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

const char DEVICE_NAME[] = "SmartCheck-TEST101";
const char SERVICE_UUID[] = "7c6b1000-9f3a-4b27-8d15-6e2a90c4f801";
const char ROOM_UUID[] = "7c6b1001-9f3a-4b27-8d15-6e2a90c4f801";
const char ROOM_ID[] = "TEST-101";
constexpr int OLED_HEIGHT = 64;
Adafruit_SSD1306 display(128, OLED_HEIGHT, &Wire, -1);
BLEAdvertising *advertising = nullptr;
std::atomic<bool> connected{false};
std::atomic<bool> restartAdvertising{false};
std::atomic<bool> roomRead{false};
// 0=off, 1=starting, 2=on (GAP confirmed), 3=error
std::atomic<int> advState{0};
std::atomic<uint32_t> totalReads{0};
bool oledReady = false;

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
ServerEvents serverEvents;
RoomEvents roomEvents;

void startAdvertising() {
  advState.store(1);
  if (!advertising->start()) advState.store(3);
}

void drawStatus() {
  const bool client = connected.load();
  const int state = advState.load();
  const char *adv = client ? "OFF (busy)" :
      state == 2 ? "ON" : state == 1 ? "STARTING" : state == 3 ? "ERROR" : "OFF";
  Serial.printf("name=%s advertising=%s connected=%s room_read=%s reads=%lu\n",
                DEVICE_NAME, adv, client ? "YES" : "NO",
                roomRead.load() ? "YES" : "NO", (unsigned long)totalReads.load());
  if (!oledReady) return;
  display.clearDisplay();
  display.setCursor(0, 0);
  display.println(DEVICE_NAME);
  display.print("ADV: "); display.println(adv);
  display.print("Client: "); display.println(client ? "CONNECTED" : "NONE");
  display.println(roomRead.load() ? "Read: TEST-101 OK" : "Read: waiting");
  if (OLED_HEIGHT >= 64) {
    display.println();
    display.println("Room: TEST-101");
    display.println("BLE only / no WiFi");
  }
  display.display();
}

void setup() {
  Serial.begin(115200);
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
