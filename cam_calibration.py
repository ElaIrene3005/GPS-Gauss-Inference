import cv2
import cv2.aruco as aruco
import glob
import os
import struct
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy


# Write COLMAP binary models directly
def write_next_bytes(fid, data, format_str):
    fid.write(struct.pack(format_str, *data))


def write_cameras_text_to_binary(txt_path, bin_path):
    cameras = {}

    with open(txt_path, "r") as fid:
        for line in fid:
            if line.startswith("#") or not line.strip():
                continue

            elems = line.split()
            camera_id = int(elems[0])
            model = elems[1]
            width = int(elems[2])
            height = int(elems[3])
            params = np.array(tuple(map(float, elems[4:])))

            cameras[camera_id] = (
                camera_id, model, width, height, params
            )

    with open(bin_path, "wb") as fid:
        write_next_bytes(fid, (len(cameras),), "Q")

        for camera_id, cam_data in cameras.items():
            model_id = (
                1 if cam_data[1] == "SIMPLE_PINHOLE"
                else 4 if cam_data[1] == "OPENCV"
                else 0
            )

            write_next_bytes(
                fid,
                (cam_data[0], model_id, cam_data[2], cam_data[3]),
                "iiQQ"
            )

            write_next_bytes(
                fid,
                tuple(cam_data[4]),
                f"{len(cam_data[4])}d"
            )


def write_images_text_to_binary(txt_path, bin_path):
    images = {}

    with open(txt_path, "r") as fid:
        lines = fid.readlines()
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            if line.startswith("#") or not line:
                i += 1
                continue

            elems = line.split()

            image_id = int(elems[0])
            qvec = np.array(tuple(map(float, elems[1:5])))
            tvec = np.array(tuple(map(float, elems[5:8])))
            camera_id = int(elems[8])
            name = elems[9]

            i += 1

            images[image_id] = (
                image_id,
                qvec,
                tvec,
                camera_id,
                name,
                np.zeros((0, 3))
            )

            i += 1

    with open(bin_path, "wb") as fid:
        write_next_bytes(fid, (len(images),), "Q")

        for image_id, img_data in images.items():
            write_next_bytes(fid, (img_data[0],), "i")
            write_next_bytes(fid, tuple(img_data[1]), "4d")
            write_next_bytes(fid, tuple(img_data[2]), "3d")
            write_next_bytes(fid, (img_data[3],), "i")

            name_bytes = img_data[4].encode("utf-8")

            write_next_bytes(
                fid,
                (name_bytes,),
                f"{len(name_bytes)}s"
            )

            fid.write(b"\x00")

            write_next_bytes(fid, (0,), "Q")


def main():

    # 1. CHARUCO BOARD CONFIGURATION
    squaresX, squaresY = 7, 5

    squareLength = 0.022
    markerLength = 0.016

    dictionary = aruco.getPredefinedDictionary(
        aruco.DICT_6X6_250
    )

    board = aruco.CharucoBoard(
        (squaresX, squaresY),
        squareLength,
        markerLength,
        dictionary
    )

    charuco_detector = aruco.CharucoDetector(board)

    # 2. IMAGE FOLDERS
    dir0 = (
        "/home/rizalka/Documents/gps-gauss+/"
        "inference_demo/colmap_workspace/images/cal_cam0"
    )

    dir1 = (
        "/home/rizalka/Documents/gps-gauss+/"
        "inference_demo/colmap_workspace/images/cal_cam1"
    )

    img_paths0 = sorted(
        glob.glob(os.path.join(dir0, "*.jpg"))
    )

    img_paths1 = sorted(
        glob.glob(os.path.join(dir1, "*.jpg"))
    )

    if not img_paths0 or not img_paths1:
        print("Error: Images not found in directories.")
        return

    if len(img_paths0) != len(img_paths1):
        print("Error: Camera image counts do not match.")
        return

    objPoints = []
    imgPoints0 = []
    imgPoints1 = []

    image_size = None

    print(
        f"Processing {len(img_paths0)} image pairs "
        "with 7x5 ChArUco layout..."
    )

    # 3. DETECT CHARUCO CORNERS
    for p0, p1 in zip(img_paths0, img_paths1):

        img0 = cv2.imread(p0)
        img1 = cv2.imread(p1)

        if img0 is None or img1 is None:
            continue

        if img0.shape[:2] != img1.shape[:2]:
            print(f"Skipping mismatched image sizes: {p0}, {p1}")
            continue

        gray0 = cv2.cvtColor(
            img0, cv2.COLOR_BGR2GRAY
        )

        gray1 = cv2.cvtColor(
            img1, cv2.COLOR_BGR2GRAY
        )

        image_size = gray0.shape[::-1]

        c_corners0, c_ids0, m_corners0, m_ids0 = (
            charuco_detector.detectBoard(gray0)
        )

        c_corners1, c_ids1, m_corners1, m_ids1 = (
            charuco_detector.detectBoard(gray1)
        )

        deb0 = img0.copy()
        deb1 = img1.copy()

        # Camera 0 visualization
        try:
            if m_ids0 is not None and len(m_ids0) > 0:
                aruco.drawDetectedMarkers(
                    deb0, m_corners0, m_ids0
                )

            if (
                c_corners0 is not None
                and c_ids0 is not None
                and len(c_ids0) > 0
                and len(c_corners0) == len(c_ids0)
            ):
                aruco.drawDetectedCornersCharuco(
                    deb0,
                    c_corners0,
                    c_ids0,
                    (0, 0, 255)
                )

        except Exception as e:
            print(f"Cam0 visualization warning: {e}")

        # Camera 1 visualization
        try:
            if m_ids1 is not None and len(m_ids1) > 0:
                aruco.drawDetectedMarkers(
                    deb1, m_corners1, m_ids1
                )

            if (
                c_corners1 is not None
                and c_ids1 is not None
                and len(c_ids1) > 0
                and len(c_corners1) == len(c_ids1)
            ):
                aruco.drawDetectedCornersCharuco(
                    deb1,
                    c_corners1,
                    c_ids1,
                    (0, 0, 255)
                )

        except Exception as e:
            print(f"Cam1 visualization warning: {e}")

        # Match common ChArUco IDs
        if (
            c_corners0 is not None
            and c_corners1 is not None
            and c_ids0 is not None
            and c_ids1 is not None
            and len(c_ids0) > 0
            and len(c_ids1) > 0
            and len(c_corners0) == len(c_ids0)
            and len(c_corners1) == len(c_ids1)
        ):

            ids0 = c_ids0.flatten()
            ids1 = c_ids1.flatten()

            common = np.intersect1d(ids0, ids1)

            if len(common) >= 6:

                idx0 = [
                    np.where(ids0 == cid)[0][0]
                    for cid in common
                ]

                idx1 = [
                    np.where(ids1 == cid)[0][0]
                    for cid in common
                ]

                objPoints.append(
                    board.getChessboardCorners()[common]
                )

                imgPoints0.append(
                    c_corners0[idx0]
                )

                imgPoints1.append(
                    c_corners1[idx1]
                )

        # Side-by-side preview
        preview = cv2.hconcat([
            cv2.resize(deb0, (640, 360)),
            cv2.resize(deb1, (640, 360))
        ])

        cv2.imshow(
            "Calibration View (Cam0 | Cam1)",
            preview
        )

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    if not objPoints:
        print("Error: No valid overlapping frame pairs found.")
        return

   # 4. INDIVIDUAL CAMERA CALIBRATION
    print(
        f"\nSuccessfully matched {len(objPoints)} pairs."
    )

    print("Calibrating Camera 0 and Camera 1...")

    # Standard 5-coefficient distortion model:
    # k1, k2, p1, p2, k3

    rms0, mtx0, dist0, _, _ = cv2.calibrateCamera(
        objPoints,
        imgPoints0,
        image_size,
        None,
        None
    )

    rms1, mtx1, dist1, _, _ = cv2.calibrateCamera(
        objPoints,
        imgPoints1,
        image_size,
        None,
        None
    )

    # 5. STEREO CALIBRATION
    print("Running stereo calibration...")

    stereo_rms, mtx0, dist0, mtx1, dist1, R, T, E, F = (
        cv2.stereoCalibrate(
            objPoints,
            imgPoints0,
            imgPoints1,
            mtx0,
            dist0,
            mtx1,
            dist1,
            image_size,
            criteria=(
                cv2.TERM_CRITERIA_EPS
                + cv2.TERM_CRITERIA_MAX_ITER,
                100,
                1e-5
            ),
            flags=cv2.CALIB_FIX_INTRINSIC
        )
    )

    # 6. CALCULATE ROTATION ANGLES
    rotation = R_scipy.from_matrix(R)

    euler_angles = rotation.as_euler(
        'xyz',
        degrees=True
    )

    # 7. PRINT CALIBRATION RESULTS
    np.set_printoptions(
        precision=6,
        suppress=True
    )

    print("\n" + "=" * 60)
    print("CHARUCO STEREO CALIBRATION RESULTS")
    print("=" * 60)

    print(f"\nCamera 0 RMS: {rms0:.6f} pixels")
    print(f"Camera 1 RMS: {rms1:.6f} pixels")
    print(f"Stereo RMS:   {stereo_rms:.6f} pixels")

    print("\n--- Camera 0 Intrinsic Matrix (K0) ---")
    print(mtx0)

    print("\n--- Camera 1 Intrinsic Matrix (K1) ---")
    print(mtx1)

    print("\n--- Camera 0 Distortion ---")
    print(dist0.ravel())

    print("\n--- Camera 1 Distortion ---")
    print(dist1.ravel())

    print("\n--- Stereo Rotation Matrix (R) ---")
    print(R)

    print("\n--- Stereo Translation Vector (T) ---")
    print(T.ravel())

    print("\n--- Relative Rotation Angles (XYZ) ---")
    print(f"Rotation X: {euler_angles[0]:.4f} degrees")
    print(f"Rotation Y: {euler_angles[1]:.4f} degrees")
    print(f"Rotation Z: {euler_angles[2]:.4f} degrees")

    print("\n--- Baseline Distance ---")
    print(f"{np.linalg.norm(T):.6f} meters")
    print(f"{np.linalg.norm(T) * 100:.2f} centimeters")

    print("\n" + "=" * 60)

    # 8. COLMAP EXPORT
    d0 = dist0.ravel()
    d1 = dist1.ravel()

    output_sparse_dir = (
        "/home/rizalka/Documents/gps-gauss+/"
        "inference_demo/raw_custom_data/sparse/0"
    )

    os.makedirs(
        output_sparse_dir,
        exist_ok=True
    )

    cameras_txt_path = os.path.join(
        output_sparse_dir, "cameras.txt"
    )

    images_txt_path = os.path.join(
        output_sparse_dir, "images.txt"
    )

    cameras_bin_path = os.path.join(
        output_sparse_dir, "cameras.bin"
    )

    images_bin_path = os.path.join(
        output_sparse_dir, "images.bin"
    )

    points3d_bin_path = os.path.join(
        output_sparse_dir, "points3D.bin"
    )

    # Write cameras.txt
    # COLMAP OPENCV order:
    # fx, fy, cx, cy, k1, k2, p1, p2

    with open(cameras_txt_path, "w") as f:

        f.write(
            "# Camera list with one line of data per camera:\n"
        )

        f.write(
            "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        )

        f.write(
            f"1 OPENCV {image_size[0]} {image_size[1]} "
            f"{mtx0[0,0]:.6f} {mtx0[1,1]:.6f} "
            f"{mtx0[0,2]:.6f} {mtx0[1,2]:.6f} "
            f"{d0[0]:.6f} {d0[1]:.6f} "
            f"{d0[2]:.6f} {d0[3]:.6f}\n"
        )

        f.write(
            f"2 OPENCV {image_size[0]} {image_size[1]} "
            f"{mtx1[0,0]:.6f} {mtx1[1,1]:.6f} "
            f"{mtx1[0,2]:.6f} {mtx1[1,2]:.6f} "
            f"{d1[0]:.6f} {d1[1]:.6f} "
            f"{d1[2]:.6f} {d1[3]:.6f}\n"
        )

    # Write images.txt
    r_scipy = R_scipy.from_matrix(R)

    quat = r_scipy.as_quat()

    qw, qx, qy, qz = (
        quat[3],
        quat[0],
        quat[1],
        quat[2]
    )

    tx, ty, tz = T.ravel()

    with open(images_txt_path, "w") as f:

        f.write(
            "# Image list with two lines of data per image:\n"
        )

        f.write(
            "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        )

        f.write(
            "# POINTS2D[] as (X, Y, POINT3D_ID)\n"
        )

        p0 = img_paths0[0]
        p1 = img_paths1[0]

        base_name0 = os.path.basename(p0)

        frame_num = ''.join(
            filter(str.isdigit, base_name0)
        )

        if not frame_num:
            frame_num = "0000"

        name0 = f"{frame_num}_cam0.jpg"
        name1 = f"{frame_num}_cam1.jpg"

        # Camera 0: Reference camera
        f.write(
            f"1 1.0 0.0 0.0 0.0 "
            f"0.0 0.0 0.0 1 {name0}\n\n"
        )

        # Camera 1: Relative pose from stereo calibration
        f.write(
            f"2 {qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f} "
            f"{tx:.6f} {ty:.6f} {tz:.6f} "
            f"2 {name1}\n\n"
        )

    # Convert text files to binary
    write_cameras_text_to_binary(
        cameras_txt_path,
        cameras_bin_path
    )

    write_images_text_to_binary(
        images_txt_path,
        images_bin_path
    )

    # Create empty points3D.bin if necessary
    if not os.path.exists(points3d_bin_path):

        with open(points3d_bin_path, "wb") as f:
            f.write(struct.pack("Q", 0))

    print(
        "\nSuccessfully generated COLMAP files in:"
    )

    print(output_sparse_dir)


if __name__ == "__main__":
    main()