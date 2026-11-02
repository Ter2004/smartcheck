# แอป SmartCheck สำหรับ Android

แอปนี้ **ครอบเว็บ SmartCheck เดิม** (หน้า log in, หน้าเช็คชื่อ, สแกนหน้า ใช้ของเว็บทั้งหมด)
สิ่งที่แอปเพิ่มให้มีอย่างเดียว: **หาสัญญาณบอร์ดในห้องเองโดยไม่ต้องกดปุ่ม** และทุกคนสแกนพร้อมกันได้ ไม่มีคิว

- บอร์ด: `firmware/smartcheck_broadcast` (ประกาศรหัสที่เปลี่ยนทุก 5 วินาที ไม่รับการเชื่อมต่อ)
- หน้าเว็บที่สแกน: `app/static/js/broadcast_room_scanner.js` + `checkin_flow.js` (ทำงานเฉพาะเมื่อเปิดในแอป)
- เซิร์ฟเวอร์ตรวจรหัส: `app/services/broadcast_beacon.py`
- iPhone / โน้ตบุ๊ก: ใช้รหัสหรือ QR บนจออาจารย์ (`CHECKIN_SESSION_CODE=1`)

## ก่อนใช้จริง (ฝั่งเซิร์ฟเวอร์)
1. apply `database/migrations/20261027_broadcast_sightings.sql` บน Supabase
2. `.env`: `CHECKIN_PROXIMITY_METHOD=broadcast` และ (แนะนำ) `CHECKIN_SESSION_CODE=1` ให้ iPhone/โน้ตบุ๊ก
3. หน้า admin → อุปกรณ์ห้องเรียน: รหัสห้อง (`ble_room_code`) ต้องตรงกับ `ROOM_ID` ในบอร์ด (ตอนนี้ `TEST-101`)

## สร้างไฟล์ APK
ต้องมี: Node, Android SDK (`%LOCALAPPDATA%\Android\Sdk`), Java ของ Android Studio — ติดตั้งไว้แล้วบนเครื่องนี้

```
cd android-app
npm install            (ครั้งแรกครั้งเดียว)
node build-apk.js https://ที่อยู่เซิร์ฟเวอร์
```
ได้ไฟล์ `android-app/dist/SmartCheck-debug.apk`

**สำคัญ:** ที่อยู่เซิร์ฟเวอร์ถูกฝังในแอปตอน build (Capacitor ยอมให้เฉพาะหน้าเว็บของที่อยู่นี้ใช้ Bluetooth)
ถ้าที่อยู่เปลี่ยน — เช่นเปิด cloudflared quick tunnel ใหม่ — ต้อง build ใหม่ (ประมาณ 1–2 นาที) แล้วลงแอปทับ
ก่อนแจกนักศึกษาควรมีที่อยู่ถาวร (Cloudflare named tunnel + โดเมน หรือ deploy บน cloud)

## แจ้งเตือนเปิดคาบ (Firebase)
วาง `google-services.json` จาก Firebase ไว้ที่ `android-app/google-services.json` ก่อน build
บรรทัดท้ายของ build จะขึ้น `notifications: on` (ไม่มีไฟล์ = `off` แอปใช้ได้ปกติแต่ไม่มีแจ้งเตือน)
ขั้นตอนทั้งหมด: `docs/push-notifications.md`

## ลงแอปบนมือถือ
- **ผ่านสาย USB:** เปิด "ตัวเลือกสำหรับนักพัฒนา" → "USB debugging" บนมือถือ แล้ว
  `%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe install -r dist\SmartCheck-debug.apk`
- **ส่งไฟล์:** ส่ง APK ให้มือถือ (เช่น Google Drive) → เปิดไฟล์ → อนุญาต "ติดตั้งแอปที่ไม่รู้จัก"

## ครั้งแรกที่เปิดหน้าเช็คชื่อ แอปจะขอสิทธิ์
- **อุปกรณ์ใกล้เคียง (Bluetooth)** และ **ตำแหน่ง** — ต้องอนุญาตทั้งคู่
  (ใช้ตำแหน่งเพราะ Android ซ่อนสัญญาณแบบ beacon ถ้าแอปบอกว่าไม่ใช้ตำแหน่ง)
- **กล้อง** — สำหรับสแกนหน้า
- Android บางรุ่นต้อง **เปิด Location ของเครื่อง** ด้วย ไม่อย่างนั้นสแกนไม่เจอ (แอปจะบอก)

## ทดสอบ (ยังไม่ได้ทดสอบบนมือถือจริง)
1. เสียบไฟบอร์ด → เปิดแอป → log in นักศึกษา → เลือกคาบ
2. หน้าเช็คชื่อควรขึ้น "กำลังหาสัญญาณห้องเรียน..." เอง แล้ว "พบสัญญาณห้อง TEST-101" ภายในไม่กี่วินาที
3. กล้องเปิดเอง → สแกนหน้า → เช็คชื่อสำเร็จ
4. จับเวลาตั้งแต่เปิดหน้าเช็คชื่อจนสำเร็จ (ติดป้าย **[วัด]** ในรายงาน)

ถ้าสแกนไม่เจอ: Bluetooth เปิดไหม / Location เปิดไหม / สิทธิ์ครบไหม / บอร์ดมีไฟ ไม่รีสตาร์ทวน (ดู `docs/broadcast_ble.md`)
