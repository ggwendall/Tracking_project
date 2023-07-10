import cv2
import mediapipe as mp
from mediapipe.python.solutions import drawing_utils as mp_drawing
import tkinter as tk


def toggle_hands():
    global enable_hands
    enable_hands = not enable_hands


def toggle_face_detection():
    global enable_face_detection
    enable_face_detection = not enable_face_detection


def toggle_face_mesh():
    global enable_face_mesh
    enable_face_mesh = not enable_face_mesh


def update_threshold(value):
    global threshold
    threshold = float(value)


def main():
    mp_hands = mp.solutions.hands
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh

    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)

    # Création d'une fenêtre pour afficher les résultats
    cv2.namedWindow('Full Tracking')
    cv2.moveWindow('Full Tracking', 100, 100)

    # Création d'une fenêtre pour afficher le cadre d'angle de la caméra
    cv2.namedWindow('Camera Frame')
    cv2.moveWindow('Camera Frame', 800, 100)

    # Initialisation des variables pour stocker les dernières positions des repères
    hand_landmarks_prev = None
    face_landmarks_prev_list = []

    # Variables de contrôle
    enable_hands = True
    enable_face_detection = True
    enable_face_mesh = True
    threshold = 0.1

    # Création de l'interface utilisateur
    root = tk.Tk()
    root.title("Paramètres")
    root.geometry("300x200")

    # Boutons de contrôle des fonctionnalités
    hands_button = tk.Button(root, text="Toggle Hands", command=toggle_hands)
    hands_button.pack()

    face_detection_button = tk.Button(root, text="Toggle Face Detection", command=toggle_face_detection)
    face_detection_button.pack()

    face_mesh_button = tk.Button(root, text="Toggle Face Mesh", command=toggle_face_mesh)
    face_mesh_button.pack()

    # Curseur pour ajuster le seuil
    threshold_label = tk.Label(root, text="Threshold")
    threshold_label.pack()

    threshold_scale = tk.Scale(root, from_=0, to=1, resolution=0.1, orient=tk.HORIZONTAL, command=update_threshold)
    threshold_scale.set(threshold)
    threshold_scale.pack()

    def close_window():
        root.destroy()

    close_button = tk.Button(root, text="Close", command=close_window)
    close_button.pack()

    def update_gui():
        root.update()

    while cv2.getWindowProperty('Full Tracking', cv2.WND_PROP_VISIBLE) >= 1:
        ret, frame = cap.read()
        if not ret:
            break

        # Inversion horizontale de la vidéo
        frame = cv2.flip(frame, 1)

        # Ajustement de la luminosité et du contraste
        alpha = 1.2  # Facteur de luminosité (>1 pour augmenter la luminosité)
        beta = 10    # Facteur de contraste (positif ou négatif)
        adjusted_frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        image = cv2.cvtColor(adjusted_frame, cv2.COLOR_BGR2RGB)

        # Détection des mains
        if enable_hands:
            results_hands = hands.process(image)
        else:
            results_hands = None

        # Détection des visages
        if enable_face_detection:
            results_face_detection = face_detection.process(image)
        else:
            results_face_detection = None

        # Suivi des points du visage
        if enable_face_mesh:
            results_face_mesh = face_mesh.process(image)
        else:
            results_face_mesh = None

        # Affichage des résultats pour les mains
        if results_hands and results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Stockage des dernières positions des repères des mains
                hand_landmarks_prev = hand_landmarks

        # Affichage des résultats pour les visages
        if results_face_detection and results_face_detection.detections:
            for detection in results_face_detection.detections:
                mp_drawing.draw_detection(image, detection)

        if results_face_mesh and results_face_mesh.multi_face_landmarks:
            # Effacez la liste précédente des dernières positions des repères du visage
            face_landmarks_prev_list.clear()

            for face_landmarks in results_face_mesh.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1),
                )

                # Stockez les dernières positions des repères du visage dans la liste
                face_landmarks_prev_list.append(face_landmarks)

        # Affichage du cadre d'angle de la caméra
        cv2.imshow('Camera Frame', frame)

        # Affichage du résultat du tracking
        cv2.imshow('Full Tracking', image)

        cv2.waitKey(1)

        # Mise à jour de l'interface utilisateur
        update_gui()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
