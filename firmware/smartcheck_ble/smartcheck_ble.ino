// ESP32 Dev Module / Arduino ESP32 3.3.11 / built-in BLE (Bluedroid).
// Broadcast experiment: no GATT server and no client connection slots.
// Upload smartcheck_ble.ino again to restore the existing GATT check-in flow.
// Libraries: Adafruit SSD1306, Adafruit GFX, Adafruit BusIO.
#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <BLEDevice.h>
#include <BLEUtils.h>
#include <atomic>

const char SERVICE_UUID[] = "7c6b1000-9f3a-4b27-8d15-6e2a90c4f801";
const char ROOM_ID[] = "TEST-101";
constexpr int OLED_HEIGHT = 64;
Adafruit_SSD1306 display(128, OLED_HEIGHT, &Wire, -1);
std::atomic<int> advState{0}; // 0=off, 1=starting, 2=on, 3=error
bool oledReady = false;

void gapEvent(esp_gap_ble_cb_event_t event, esp_ble_gap_cb_param_t *param) {
  if (event == ESP_GAP_BLE_ADV_START_COMPLETE_EVT) {
    advState.store(param->adv_start_cmpl.status == ESP_BT_STATUS_SUCCESS ? 2 : 3);
  } else if (event == ESP_GAP_BLE_ADV_STOP_COMPLETE_EVT) {
    advState.store(param->adv_stop_cmpl.status == ESP_BT_STATUS_SUCCESS ? 0 : 3);
  }
}

void drawStatus() {
  const int state = advState.load();
  const char *label = state == 2 ? "ON" : state == 1 ? "STARTING" : state == 3 ? "ERROR" : "OFF";
  Serial.printf("beacon=SC-TEST1 room=%s advertising=%s mode=non-connectable payload=manufacturer-B\n", ROOM_ID, label);
  if (!oledReady) return;
  display.clearDisplay();
  display.setCursor(0, 0);
  display.println("SmartCheck BEACON");
  display.print("ADV: "); display.println(label);
  display.println("Room: TEST-101");
  display.println("No connections");
  if (OLED_HEIGHT >= 64) {
    display.println();
    display.println("Listeners: unknown");
    display.println("BLE only / no WiFi");
  }
  display.display();
}

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);
  Wire.setTimeOut(50);
  Wire.beginTransmission(0x3C);
  oledReady = Wire.endTransmission() == 0;
  if (oledReady) oledReady = display.begin(SSD1306_SWITCHCAPVCC, 0x3C, false, false);
  if (oledReady) {
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setTextWrap(false);
  } else {
    Serial.println("OLED unavailable; continuing BLE");
  }
  BLEDevice::init("SC-TEST1");
  BLEDevice::setCustomGapHandler(gapEvent);
  BLEAdvertising *advertising = BLEDevice::getAdvertising();
  // Scannable, NON-connectable: scan responses do not allocate a connection.
  advertising->setAdvertisementType(ADV_TYPE_SCAN_IND);
  advertising->setMinInterval(160); // 100 ms (0.625 ms units)
  advertising->setMaxInterval(240); // 150 ms
  BLEAdvertisementData primary;
  primary.setFlags(0x06); // 3 bytes
  // Hypothesis B: testing identifier 0xFFFF (little endian), then TEST-101.
  const char manufacturerPayload[] = {char(0xFF), char(0xFF), 'T', 'E', 'S', 'T', '-', '1', '0', '1'};
  primary.setManufacturerData(String(manufacturerPayload, sizeof(manufacturerPayload)));
  BLEAdvertisementData response;
  response.setCompleteServices(BLEUUID(SERVICE_UUID)); // 18 bytes
  response.setName("SC-TEST1"); // 10 bytes: scan response total = 28
  if (!advertising->setAdvertisementData(primary) || !advertising->setScanResponseData(response)) {
    advState.store(3);
  } else {
    advertising->setScanResponse(true);
    advState.store(1);
    if (!advertising->start()) advState.store(3);
  }
  drawStatus();
}

void loop() {
  static uint32_t lastDraw = 0;
  if (millis() - lastDraw >= 1000) {
    lastDraw = millis();
    drawStatus();
  }
  delay(10);
}
