from ultralytics import YOLO
import cv2 as cv
import os
import time

# Inisialisasi model YOLO


# Fungsi untuk memproses video frame-by-frame
def process_video(video_path, output_folder, start_frame, model):
    # Membuat folder output jika belum ada
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    scanms = []
    # Membuka video
    cap = cv.VideoCapture(video_path)
    frame_idx = start_frame

    # Melompat ke frame awal yang diinginkan
    cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Simpan frame sementara sebagai gambar
        temp_frame_path = f"{output_folder}/frame_{frame_idx}.jpg"
        cv.imwrite(temp_frame_path, frame)

        # Prediksi menggunakan YOLO
        start_time = time.time()
        results = model.predict(source=frame, device ="cuda:0", verbose=True, stream=False)
        scanms.append((time.time()-start_time)*1000)
        print(f"Frame {frame_idx}: {scanms[-1]:.4f} ms")
        results = model.predict(source=temp_frame_path, device ="cuda:0", verbose = False, stream=False, save = True, line_width = 2)
        frame_idx += 1
    # Hitung rata-rata waktu prediksi per frame
    avg_scan_time = sum(scanms) / len(scanms) if scanms else 0
    fps = 1000 / avg_scan_time if avg_scan_time else 0
    output_file = os.path.join(output_folder, "results.txt")

    with open(output_file, "a") as f:
        f.write(f"Average prediction time per frame: {avg_scan_time:.4f} ms\n")
        f.write(f"FPS: {fps:.2f}\n")
    cap.release()
    cv.destroyAllWindows()

# Jalankan fungsi untuk memproses video
# process_video("videoplayback.mp4", "predict", 0, YOLO('yolo11n.pt', task="detect"))
process_video("videoplayback.mp4", "predict", 0, YOLO('yolo11s.pt', task="detect"))
# process_video("videoplayback.mp4", "predict", 0, YOLO('yolo11m.pt', task="detect"))
# process_video("videoplayback.mp4", "predict", 0, YOLO('yolo11l.pt', task="detect"))
# process_video("videoplayback.mp4", "predict", 0, YOLO('yolo11x.pt', task="detect"))
