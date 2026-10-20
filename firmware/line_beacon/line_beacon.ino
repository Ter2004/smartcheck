// SmartCheck LINE Beacon board (trial): advertises a LINE Simple Beacon so the
// LINE app detects the room, with a signed counter in the device message.
//
// Packet (LINE Simple Beacon spec, github.com/line/line-simple-beacon):
//   Flags | Complete 16-bit UUIDs: 0xFE6F | Service Data 0xFE6F:
//     frame type 0x02 | HWID (5 bytes) | measured TxPower 0x7F | device message (12 bytes)
// Device message = counter (4 bytes big-endian) || HMAC-SHA256(room key,
//     "smartcheck-line-v1" || counter)[0..8)  — app/services/line_beacon.py checks it.
// No clock or WiFi: the counter starts at a random value at boot and steps every
// COUNTER_STEP_MS; the server only accepts a counter within 120 s of its first sighting.
//
// Board: any ESP32 with the Arduino core (BLEDevice). Non-connectable, so any
// number of phones hear it at once. Separate from firmware/smartcheck_ble.

#include <Arduino.h>
#include <BLEDevice.h>
#include <BLEUtils.h>
#include <ctype.h>
#include <string.h>
#include "esp_random.h"
#include "mbedtls/md.h"
#if __has_include("secret.h")
#include "secret.h"  // ROOM_SECRET_HEX (admin page) and LINE_HWID_HEX (LINE OA Manager); git-ignored
#else
#error "Missing secret.h: copy secret.h.example, then fill in ROOM_SECRET_HEX and LINE_HWID_HEX"
#endif

static const uint32_t COUNTER_STEP_MS = 10000;
static const uint8_t CONTEXT[] = "smartcheck-line-v1";
static const size_t CONTEXT_LEN = sizeof(CONTEXT) - 1;

uint8_t roomKey[32];
uint8_t hwid[5];
uint32_t counterBase = 0;
uint32_t lastCounter = 0xFFFFFFFF;
BLEAdvertising *advertising = nullptr;

bool parseHex(const char *hex, uint8_t *out, size_t len) {
  if (strlen(hex) != 2 * len) return false;
  uint8_t any = 0;
  for (size_t i = 0; i < len; i++) {
    const char pair[3] = {hex[2 * i], hex[2 * i + 1], 0};
    if (!isxdigit((unsigned char)pair[0]) || !isxdigit((unsigned char)pair[1])) return false;
    out[i] = (uint8_t)strtoul(pair, nullptr, 16);
    any |= out[i];
  }
  return any != 0;  // all-zero values from secret.h.example are refused
}

bool deviceMessage(uint32_t counter, uint8_t out[12]) {
  out[0] = counter >> 24; out[1] = counter >> 16; out[2] = counter >> 8; out[3] = counter;
  uint8_t msg[CONTEXT_LEN + 4], mac[32];
  memcpy(msg, CONTEXT, CONTEXT_LEN);
  memcpy(msg + CONTEXT_LEN, out, 4);
  const mbedtls_md_info_t *info = mbedtls_md_info_from_type(MBEDTLS_MD_SHA256);
  if (!info || mbedtls_md_hmac(info, roomKey, 32, msg, sizeof(msg), mac) != 0) return false;
  memcpy(out + 4, mac, 8);
  return true;
}

void advertise(uint32_t counter) {
  uint8_t dm[12];
  if (!deviceMessage(counter, dm)) return;
  // Service data payload after the 16-bit UUID: frame, HWID, TxPower, device message.
  uint8_t service[2 + 1 + 5 + 1 + 12] = {0x6F, 0xFE, 0x02};
  memcpy(service + 3, hwid, 5);
  service[8] = 0x7F;
  memcpy(service + 9, dm, 12);

  BLEAdvertisementData data;
  data.setFlags(0x06);  // LE general discoverable, BR/EDR not supported
  const uint8_t uuids[] = {0x03, 0x03, 0x6F, 0xFE};
  data.addData(String((const char *)uuids, sizeof(uuids)));
  uint8_t sd[2 + sizeof(service)];
  sd[0] = 1 + sizeof(service);  // length: type + payload
  sd[1] = 0x16;                 // Service Data - 16-bit UUID
  memcpy(sd + 2, service, sizeof(service));
  data.addData(String((const char *)sd, sizeof(sd)));

  advertising->stop();
  advertising->setAdvertisementData(data);
  advertising->setAdvertisementType(ADV_TYPE_NONCONN_IND);
  advertising->setMinInterval(0xA0);  // 100 ms (units of 0.625 ms)
  advertising->setMaxInterval(0xA0);
  advertising->start();
  Serial.printf("counter=%lu dm=", (unsigned long)counter);
  for (int i = 0; i < 12; i++) Serial.printf("%02x", dm[i]);
  Serial.println();
}

void setup() {
  Serial.begin(115200);
  if (!parseHex(ROOM_SECRET_HEX, roomKey, 32) || !parseHex(LINE_HWID_HEX, hwid, 5)) {
    Serial.println("KEY ERROR: set ROOM_SECRET_HEX (64 hex) and LINE_HWID_HEX (10 hex) in secret.h");
    while (true) delay(1000);
  }
  counterBase = esp_random() & 0x7FFFFFFF;  // fresh start each boot; old recordings stay old
  BLEDevice::init("SmartCheck-LINE");
  advertising = BLEDevice::getAdvertising();
}

void loop() {
  const uint32_t counter = counterBase + millis() / COUNTER_STEP_MS;
  if (counter != lastCounter) {
    advertise(counter);
    lastCounter = counter;
  }
  delay(200);
}
