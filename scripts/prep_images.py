from PIL import Image
import glob, os

for f in glob.glob('test_images/*.jpg'):
    if '_ok' in f:
        continue
    im = Image.open(f)
    w, h = im.size
    # ครอบเฉพาะบริเวณใบหน้า ตัดพื้นหลังลายออก
    im = im.crop((int(w*0.15), int(h*0.06), int(w*0.85), int(h*0.68)))
    # ย่อด้วย LANCZOS = antialiasing เต็มที่ ไม่สร้าง aliasing
    nw, nh = im.size
    scale = 900 / nh
    im = im.resize((int(nw*scale), 900), Image.LANCZOS)
    out = f.replace('.jpg', '_ok.jpg')
    im.save(out, quality=90)
    print(out, im.size, os.path.getsize(out)//1024, 'KB')