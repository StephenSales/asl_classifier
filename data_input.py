import cv2
import mediapipe as mp
import csv
from collections import defaultdict

# Labels to collect
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
current_label = 'A'
samples = []
sample_counts = defaultdict(int)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

print(f"[INFO] Starting collection for label: {current_label} (Press 'S' to switch, 'Q' to quit)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract and save landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])

            landmarks.append(current_label)
            samples.append(landmarks)
            sample_counts[current_label] += 1

    # Draw info on screen
    cv2.putText(image, f"Label: {current_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv2.putText(image, f"Samples: {sample_counts[current_label]}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow("Hand Sign Data Collection", image)

    key = cv2.waitKey(10)
    if key == ord('q'):
        break
    elif key == ord('s'):
        idx = (labels.index(current_label) + 1) % len(labels)
        current_label = labels[idx]
        print(f"[INFO] Now collecting for: {current_label}")

cap.release()
cv2.destroyAllWindows()

# Save to CSV
with open("dataset.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for row in samples:
        writer.writerow(row)

print("[INFO] Data saved to dataset.csv")
print("Samples per label:")
for label, count in sample_counts.items():
    print(f"  {label}: {count}")
