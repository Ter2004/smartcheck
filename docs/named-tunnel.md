# ลิงก์ถาวร: โดเมน + Cloudflare named tunnel

ทำครั้งเดียว (~1 ชั่วโมง รวมรอ DNS) แล้วลิงก์จะไม่เปลี่ยนอีก ไม่ว่าจะปิดเปิดโน้ตบุ๊กกี่ครั้ง
แอป Android ฝังลิงก์ไว้ตอน build ([android-app/capacitor.config.json](../android-app/capacitor.config.json)) — ลิงก์คงที่ = build APK ครั้งเดียวใช้ได้ทั้งวันทดสอบ วันนำเสนอ และใส่ในโปสเตอร์

ตัวอย่างข้างล่างใช้โดเมน `smartcheck.me` และลิงก์ `app.smartcheck.me` — เปลี่ยนเป็นโดเมนที่ซื้อจริง

## 1. ซื้อโดเมน
- Namecheap (หรือที่ไหนก็ได้) · ถ้าได้ GitHub Student Pack ลองรับ .me ฟรี 1 ปีที่ nc.me ก่อน
- ดูราคา **"Renews at"** ในหน้าจ่ายเงินด้วย (ปีต่อไปมักแพงกว่าปีแรก)

## 2. ให้ Cloudflare ดูแล DNS (ฟรี)
1. สมัคร Cloudflare › **Add a domain** › ใส่ `smartcheck.me` › เลือกแพ็กเกจ **Free**
2. Cloudflare ให้ nameserver 2 ตัว (เช่น `xxx.ns.cloudflare.com`)
3. Namecheap › Domain List › Manage › **Nameservers: Custom DNS** › ใส่ 2 ตัวนั้น › บันทึก
4. รอจน Cloudflare แจ้งว่าโดเมน **Active** (ปกติไม่กี่นาที บางครั้งหลายชั่วโมง)

## 3. สร้าง tunnel บนโน้ตบุ๊ก (PowerShell ในโฟลเดอร์ `smartcheck_project`)
```
.\cloudflared.exe tunnel login
.\cloudflared.exe tunnel create smartcheck
.\cloudflared.exe tunnel route dns smartcheck app.smartcheck.me
```
- `login` เปิดเบราว์เซอร์ให้เลือกโดเมน · `create` สร้างไฟล์กุญแจ `C:\Users\<ชื่อ>\.cloudflared\<TUNNEL-ID>.json` (เก็บไว้ ห้ามแชร์ ห้าม commit)

## 4. ไฟล์ตั้งค่า `C:\Users\<ชื่อ>\.cloudflared\config.yml`
```yaml
tunnel: smartcheck
credentials-file: C:\Users\<ชื่อ>\.cloudflared\<TUNNEL-ID>.json
ingress:
  - hostname: app.smartcheck.me
    service: http://127.0.0.1:5000
  - service: http_status:404
```
ใช้ `127.0.0.1` ไม่ใช่ `localhost`: บน Windows เครื่องนี้ `localhost` ช้ากว่า ~205 ms ต่อการเปิด connection (วัด 2026-11-07)

## 5. เปิดใช้
- ดับเบิลคลิก [scripts/field_test/start-field-test.cmd](../scripts/field_test/start-field-test.cmd) — เปิดเซิร์ฟเวอร์ + tunnel (เห็นไฟล์ config.yml แล้วจะใช้ named tunnel เอง)
- เปิด `https://app.smartcheck.me/login` จากมือถือ (ปิด Wi-Fi ลองผ่านเน็ตมือถือด้วย)

## 6. Build แอปด้วยลิงก์ใหม่ แล้วลงทับทุกเครื่อง
```
cd android-app
node build-apk.js https://app.smartcheck.me
```
ได้ `android-app/dist/SmartCheck-debug.apk` · ทุกคนต้องล็อกอินใหม่ 1 ครั้ง (การล็อกอินผูกกับชื่อเว็บ)

## ไม่บังคับ
- `PUBLIC_BASE_URL=https://app.smartcheck.me` ใน `.env` — ใช้กับ LINE Beacon เท่านั้น (ตอนนี้ปิดอยู่) · รหัสสำรอง/QR ใช้ลิงก์จาก request อยู่แล้ว
