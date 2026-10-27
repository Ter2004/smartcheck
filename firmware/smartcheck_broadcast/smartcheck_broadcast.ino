// SmartCheck broadcast board: advertises a signed rolling code and accepts no
// connections, so any number of phones can read it at the same time.
//
// Advertisement (non-connectable, legacy, 30 of 31 bytes):
//   Flags (3) | Manufacturer Specific Data (4 + 23):
//     company ID 0xFFFF (testing) | payload
// Payload (23 bytes):
//   "SC" | version 0x01 | board id (8 bytes, ble_room_code padded with 0x00)
//   | counter (4 bytes, big-endian)
//   | HMAC-SHA256(room key, "smartcheck-bcast-v1" || board id || counter)[0..8)
//
// No clock or WiFi: the counter steps every COUNTER_STEP_MS. It only moves forward,
// across reboots too: the board reserves RESERVE_STEPS counters at a time in flash
// and restarts from the end of the last reservation, so the server can refuse any
// counter older than the newest one it has seen. Only the very first boot picks a
// random start.
//
// Board: ESP32 Dev Module, Espressif Arduino core 3.3.12 (built-in BLE).

#include <Arduino.h>
#include <BLEDevice.h>
#include <Preferences.h>
#include <ctype.h>
#include <string.h>
#include "esp_random.h"
#include "mbedtls/md.h"
#if __has_include("secret.h")
#include "secret.h"  // ROOM_SECRET_HEX from the admin page; git-ignored
#else
#error "Missing secret.h: copy secret.h.example to secret.h and paste the room key from the admin page"
#endif

// Must equal beacons.ble_room_code of this room (at most 8 ASCII characters).
const char ROOM_ID[] = "TEST-101";
const char DEVICE_NAME[] = "SmartCheck-Bcast";

static const uint32_t COUNTER_STEP_MS = 5000;
static const uint32_t RESERVE_STEPS = 720;  // one flash write per hour of counters
static const uint8_t CONTEXT[] = "smartcheck-bcast-v1";  // must match the server
static const size_t CONTEXT_LEN = sizeof(CONTEXT) - 1;
static const uint16_t COMPANY_ID = 0xFFFF;  // reserved for testing by the Bluetooth SIG
static const uint8_t VERSION = 0x01;
static const size_t BOARD_ID_LEN = 8;
static const size_t MAC_LEN = 8;
static const size_t PAYLOAD_LEN = 2 + 1 + BOARD_ID_LEN + 4 + MAC_LEN;  // 23

uint8_t roomKey[32];
uint8_t boardId[BOARD_ID_LEN];
uint32_t counterBase = 0;
uint32_t reservedUntil = 0;  // first counter not yet covered by the flash reservation
uint32_t lastCounter = 0;
bool started = false;
Preferences prefs;
BLEAdvertising *advertising = nullptr;

// 64 hex chars, not all zero (secret.h.example ships zeros on purpose).
bool parseKey(const char *hex, uint8_t out[32]) {
  if (strlen(hex) != 64) return false;
  uint8_t any = 0;
  for (int i = 0; i < 32; i++) {
    const char pair[3] = {hex[2 * i], hex[2 * i + 1], 0};
    if (!isxdigit((unsigned char)pair[0]) || !isxdigit((unsigned char)pair[1])) return false;
    out[i] = (uint8_t)strtoul(pair, nullptr, 16);
    any |= out[i];
  }
  return any != 0;
}

// 1..8 printable ASCII characters, padded with 0x00 to 8 bytes.
bool parseBoardId(const char *id, uint8_t out[BOARD_ID_LEN]) {
  const size_t n = strlen(id);
  if (n == 0 || n > BOARD_ID_LEN) return false;
  memset(out, 0, BOARD_ID_LEN);
  for (size_t i = 0; i < n; i++) {
    if (id[i] < 0x21 || id[i] > 0x7E) return false;
    out[i] = (uint8_t)id[i];
  }
  return true;
}

bool buildPayload(const uint8_t key[32], const uint8_t id[BOARD_ID_LEN], uint32_t counter,
                  uint8_t out[PAYLOAD_LEN]) {
  out[0] = 'S'; out[1] = 'C'; out[2] = VERSION;
  memcpy(out + 3, id, BOARD_ID_LEN);
  uint8_t *c = out + 3 + BOARD_ID_LEN;
  c[0] = counter >> 24; c[1] = counter >> 16; c[2] = counter >> 8; c[3] = counter;
  uint8_t msg[CONTEXT_LEN + BOARD_ID_LEN + 4], mac[32];
  memcpy(msg, CONTEXT, CONTEXT_LEN);
  memcpy(msg + CONTEXT_LEN, id, BOARD_ID_LEN);
  memcpy(msg + CONTEXT_LEN + BOARD_ID_LEN, c, 4);
  const mbedtls_md_info_t *info = mbedtls_md_info_from_type(MBEDTLS_MD_SHA256);
  if (!info || mbedtls_md_hmac(info, key, 32, msg, sizeof(msg), mac) != 0) return false;
  memcpy(c + 4, mac, MAC_LEN);
  return true;
}

// Same vector as the server tests: key 00..1f, board "TEST-101", counter 0x01020304.
bool selfTest() {
  static const uint8_t expected[PAYLOAD_LEN] = {
    0x53, 0x43, 0x01, 0x54, 0x45, 0x53, 0x54, 0x2d, 0x31, 0x30, 0x31, 0x01,
    0x02, 0x03, 0x04, 0x7b, 0x69, 0x19, 0x85, 0x54, 0x29, 0xe7, 0x31};
  uint8_t key[32], id[BOARD_ID_LEN], out[PAYLOAD_LEN];
  for (int i = 0; i < 32; i++) key[i] = i;
  return parseBoardId("TEST-101", id) && buildPayload(key, id, 0x01020304, out) &&
         memcmp(out, expected, PAYLOAD_LEN) == 0;
}

// Store the end of a new reservation before any counter in it is advertised.
bool reserve(uint32_t from) {
  reservedUntil = from + RESERVE_STEPS;
  return prefs.putUInt("next", reservedUntil) == sizeof(uint32_t);
}

void advertise(uint32_t counter) {
  uint8_t payload[PAYLOAD_LEN];
  if (!buildPayload(roomKey, boardId, counter, payload)) {
    Serial.println("HMAC ERROR: advertisement not updated");
    return;
  }
  uint8_t msd[2 + 2 + PAYLOAD_LEN];
  msd[0] = 1 + 2 + PAYLOAD_LEN;  // length: type + company ID + payload
  msd[1] = 0xFF;                 // Manufacturer Specific Data
  msd[2] = COMPANY_ID & 0xFF;    // little-endian
  msd[3] = COMPANY_ID >> 8;
  memcpy(msd + 4, payload, PAYLOAD_LEN);

  BLEAdvertisementData data;
  data.setFlags(0x06);  // LE general discoverable, BR/EDR not supported
  data.addData(String((const char *)msd, sizeof(msd)));

  advertising->stop();
  advertising->setAdvertisementData(data);
  advertising->setAdvertisementType(ADV_TYPE_NONCONN_IND);
  advertising->setMinInterval(0xA0);  // 100 ms (units of 0.625 ms)
  advertising->setMaxInterval(0xA0);
  advertising->start();
  Serial.printf("counter=%lu payload=", (unsigned long)counter);
  for (size_t i = 0; i < PAYLOAD_LEN; i++) Serial.printf("%02x", payload[i]);
  Serial.println();
}

void setup() {
  Serial.begin(115200);
  delay(200);
  const bool testOk = selfTest();
  Serial.println(testOk ? "HMAC self-test PASS" : "HMAC self-test FAIL");
  if (!testOk) {
    while (true) delay(1000);
  }
  if (!parseKey(ROOM_SECRET_HEX, roomKey)) {
    Serial.println("KEY ERROR: ROOM_SECRET_HEX in secret.h must be the 64-hex room key from the admin page");
    while (true) delay(1000);
  }
  if (!parseBoardId(ROOM_ID, boardId)) {
    Serial.println("ROOM_ID ERROR: 1-8 printable ASCII characters, same as ble_room_code");
    while (true) delay(1000);
  }

  prefs.begin("sc-bcast", false);
  if (prefs.isKey("next")) {
    counterBase = prefs.getUInt("next");
    Serial.printf("counter resumes at %lu (saved in flash)\n", (unsigned long)counterBase);
  } else {
    counterBase = esp_random() & 0x7FFFFFFF;  // first boot only; headroom before 2^32
    Serial.printf("first boot: random counter start %lu\n", (unsigned long)counterBase);
  }
  if (!reserve(counterBase)) {
    // Without a saved reservation a reboot could repeat counters; the server would
    // refuse them, so stop rather than broadcast codes that will not work later.
    Serial.println("FLASH ERROR: could not save the counter reservation");
    while (true) delay(1000);
  }

  BLEDevice::init(DEVICE_NAME);
  advertising = BLEDevice::getAdvertising();
  Serial.printf("board=%s step=%lums non-connectable\n", ROOM_ID, (unsigned long)COUNTER_STEP_MS);
}

void loop() {
  const uint32_t counter = counterBase + millis() / COUNTER_STEP_MS;
  if (!started || counter != lastCounter) {
    if (counter >= reservedUntil && !reserve(counter)) {
      Serial.println("FLASH ERROR: could not extend the counter reservation; advertising stopped");
      advertising->stop();
      while (true) delay(1000);
    }
    advertise(counter);
    lastCounter = counter;
    started = true;
  }
  delay(100);
}
