// ESP32 Dev Module, Espressif Arduino core 3.3.11 (built-in BLE/Bluedroid).
// Libraries: Adafruit SSD1306, Adafruit GFX, Adafruit BusIO.
// SSD1306 assumed 128x64. For a 128x32 panel change OLED_HEIGHT to 32.
#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <atomic>
#include <ctype.h>
#include <string.h>
#include "mbedtls/md.h"
#if __has_include("secret.h")
#include "secret.h"  // ROOM_SECRET_HEX from the admin page; git-ignored
#else
#error "Missing secret.h: admin page -> อุปกรณ์ห้องเรียน -> สร้าง secret, save it as secret.h next to this sketch"
#endif

const char DEVICE_NAME[] = "SmartCheck-TEST101";
const char SERVICE_UUID[] = "7c6b1000-9f3a-4b27-8d15-6e2a90c4f801";
const char ROOM_UUID[] = "7c6b1001-9f3a-4b27-8d15-6e2a90c4f801";
// Write a 16-byte server nonce, then read HMAC-SHA256(room key, context + nonce).
const char CHALLENGE_UUID[] = "7c6b1002-9f3a-4b27-8d15-6e2a90c4f801";
const char ROOM_ID[] = "TEST-101";
const uint8_t HMAC_CONTEXT[] = "smartcheck-ble-v1";  // must match app/services/ble_challenge.py
constexpr size_t CONTEXT_LEN = sizeof(HMAC_CONTEXT) - 1;  // without the NUL
constexpr size_t NONCE_LEN = 16;
constexpr size_t MAC_LEN = 32;
constexpr int OLED_HEIGHT = 64;
Adafruit_SSD1306 display(128, OLED_HEIGHT, &Wire, -1);
BLEAdvertising *advertising = nullptr;
BLECharacteristic *challenge = nullptr;
std::atomic<bool> connected{false};
std::atomic<bool> restartAdvertising{false};
std::atomic<bool> roomRead{false};
// 0=off, 1=starting, 2=on (GAP confirmed), 3=error
std::atomic<int> advState{0};
std::atomic<uint32_t> totalReads{0};
std::atomic<uint32_t> totalSigns{0};
bool oledReady = false;
uint8_t roomKey[32];
bool keyReady = false;
bool selfTestOk = false;

bool signNonce(const uint8_t *key, const uint8_t *nonce, uint8_t out[MAC_LEN]) {
  uint8_t msg[CONTEXT_LEN + NONCE_LEN];
  memcpy(msg, HMAC_CONTEXT, CONTEXT_LEN);
  memcpy(msg + CONTEXT_LEN, nonce, NONCE_LEN);
  const mbedtls_md_info_t *info = mbedtls_md_info_from_type(MBEDTLS_MD_SHA256);
  return info && mbedtls_md_hmac(info, key, 32, msg, sizeof(msg), out) == 0;
}

// 64 hex chars, not all zero (secret.h.example ships zeros on purpose).
bool parseKey(const char *hex, uint8_t out[32]) {
  if (strlen(hex) != 64) return false;
  uint8_t any = 0;
  for (int i = 0; i < 32; i++) {
    const char hi = hex[2 * i], lo = hex[2 * i + 1];
    if (!isxdigit((unsigned char)hi) || !isxdigit((unsigned char)lo)) return false;
    const char pair[3] = {hi, lo, 0};
    out[i] = (uint8_t)strtoul(pair, nullptr, 16);
    any |= out[i];
  }
  return any != 0;
}

// Same vector as tests/test_ble_challenge.py: key 00..1f, nonce 00..0f.
bool selfTest() {
  static const uint8_t expected[MAC_LEN] = {
    0x97, 0x5f, 0xe6, 0xe2, 0x37, 0xc2, 0x3b, 0x0e, 0x5b, 0x12, 0x63, 0x78, 0x9a, 0x36, 0x46, 0x16,
    0x1a, 0x57, 0x76, 0x3c, 0x3d, 0x51, 0x2d, 0xed, 0x8c, 0x7e, 0x79, 0x7f, 0x50, 0x29, 0xf4, 0x8a};
  uint8_t key[32], nonce[NONCE_LEN], mac[MAC_LEN];
  for (int i = 0; i < 32; i++) key[i] = i;
  for (int i = 0; i < (int)NONCE_LEN; i++) nonce[i] = i;
  return signNonce(key, nonce, mac) && memcmp(mac, expected, MAC_LEN) == 0;
}

void gapEvent(esp_gap_ble_cb_event_t event, esp_ble_gap_cb_param_t *param) {
  if (event == ESP_GAP_BLE_ADV_START_COMPLETE_EVT) {
    advState.store(param->adv_start_cmpl.status == ESP_BT_STATUS_SUCCESS ? 2 : 3);
  } else if (event == ESP_GAP_BLE_ADV_STOP_COMPLETE_EVT) {
    advState.store(param->adv_stop_cmpl.status == ESP_BT_STATUS_SUCCESS ? 0 : 3);
  }
}

class ServerEvents : public BLEServerCallbacks {
  void onConnect(BLEServer *) override {
    connected.store(true);
    roomRead.store(false);
    // A new client must not read the previous client's answer.
    if (challenge) challenge->setValue((uint8_t *)"", 0);
    // Legacy connectable advertising ends on connection. One client at a time.
    advState.store(0);
  }
  void onDisconnect(BLEServer *) override {
    connected.store(false);
    restartAdvertising.store(true);
  }
};

class RoomEvents : public BLECharacteristicCallbacks {
  void onRead(BLECharacteristic *) override {
    roomRead.store(true);
    totalReads.fetch_add(1);
  }
};

class ChallengeEvents : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *c) override {
    uint8_t mac[MAC_LEN];
    // Anything but a 16-byte nonce leaves nothing to read.
    if (keyReady && c->getLength() == NONCE_LEN && signNonce(roomKey, c->getData(), mac)) {
      c->setValue(mac, MAC_LEN);
      totalSigns.fetch_add(1);
    } else {
      c->setValue((uint8_t *)"", 0);
    }
  }
};
ServerEvents serverEvents;
RoomEvents roomEvents;
ChallengeEvents challengeEvents;

void startAdvertising() {
  advState.store(1);
  if (!advertising->start()) advState.store(3);
}

const char *keyStatus() {
  return !keyReady ? "KEY ERROR" : !selfTestOk ? "HMAC ERROR" : nullptr;
}

void drawStatus() {
  const bool client = connected.load();
  const int state = advState.load();
  const char *adv = client ? "OFF (busy)" :
      state == 2 ? "ON" : state == 1 ? "STARTING" : state == 3 ? "ERROR" : "OFF";
  const char *problem = keyStatus();
  // Never print the key: only whether it loaded.
  Serial.printf("name=%s advertising=%s connected=%s room_read=%s reads=%lu signs=%lu key=%s\n",
                DEVICE_NAME, adv, client ? "YES" : "NO",
                roomRead.load() ? "YES" : "NO", (unsigned long)totalReads.load(),
                (unsigned long)totalSigns.load(), problem ? problem : "OK");
  if (!oledReady) return;
  display.clearDisplay();
  display.setCursor(0, 0);
  display.println(DEVICE_NAME);
  display.print("ADV: "); display.println(adv);
  display.print("Client: "); display.println(client ? "CONNECTED" : "NONE");
  display.println(roomRead.load() ? "Read: TEST-101 OK" : "Read: waiting");
  if (OLED_HEIGHT >= 64) {
    if (problem) display.println(problem);
    else { display.print("Signed: "); display.println((unsigned long)totalSigns.load()); }
    display.println("Room: TEST-101");
    display.println("BLE only / no WiFi");
  }
  display.display();
}

void setup() {
  Serial.begin(115200);
  keyReady = parseKey(ROOM_SECRET_HEX, roomKey);
  selfTestOk = selfTest();
  Serial.println(selfTestOk ? "HMAC self-test PASS" : "HMAC self-test FAIL");
  if (!keyReady) Serial.println("secret.h: ROOM_SECRET_HEX must be 64 hex chars from the admin page");

  Wire.begin(21, 22);
  Wire.setTimeOut(50);
  // Probe address as begin() alone does not reliably detect an absent panel.
  Wire.beginTransmission(0x3C);
  oledReady = Wire.endTransmission() == 0;
  if (oledReady) oledReady = display.begin(SSD1306_SWITCHCAPVCC, 0x3C, false, false);
  if (oledReady) {
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setTextWrap(false);
    display.clearDisplay();
    display.setCursor(0, 0);
    display.println(DEVICE_NAME);
    display.println("BLE starting...");
    display.display();
  } else {
    Serial.println("OLED unavailable at 0x3C; continuing BLE without display");
  }

  BLEDevice::init(DEVICE_NAME);
  BLEDevice::setCustomGapHandler(gapEvent);
  BLEServer *server = BLEDevice::createServer();
  server->setCallbacks(&serverEvents);
  BLEService *service = server->createService(SERVICE_UUID);
  BLECharacteristic *room = service->createCharacteristic(ROOM_UUID, BLECharacteristic::PROPERTY_READ);
  room->setValue(ROOM_ID); // Exactly 8 ASCII bytes, no newline or terminating NUL.
  room->setCallbacks(&roomEvents);
  challenge = service->createCharacteristic(
      CHALLENGE_UUID, BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_WRITE);
  challenge->setValue((uint8_t *)"", 0);
  challenge->setCallbacks(&challengeEvents);
  service->start();

  advertising = BLEDevice::getAdvertising();
  // Keep the 128-bit service UUID in the primary 31-byte advertisement.
  // Put the full name in the separate scan response to avoid truncation.
  BLEAdvertisementData primary;
  primary.setFlags(0x06);
  primary.setCompleteServices(BLEUUID(SERVICE_UUID));
  BLEAdvertisementData scanResponse;
  scanResponse.setName(DEVICE_NAME);
  if (!advertising->setAdvertisementData(primary) ||
      !advertising->setScanResponseData(scanResponse)) {
    advState.store(3);
    Serial.println("BLE advertising data configuration failed; reset board");
  } else {
    advertising->setScanResponse(true);
    startAdvertising();
  }
  drawStatus();
}

void loop() {
  // No I2C/display work in BLE callbacks; all rendering stays in this task.
  if (restartAdvertising.exchange(false)) {
    delay(150);
    if (!connected.load()) startAdvertising();
  }
  static uint32_t lastDraw = 0;
  if (millis() - lastDraw >= 250) {
    lastDraw = millis();
    drawStatus();
  }
  delay(10);
}
