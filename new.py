from ultralytics import YOLO
import cv2
import math

model = YOLO("best.pt")

classNames = ["paper", "rock", "scissors"]

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1200)
cap.set(4, 680)

# Helper function to determine the winner
def determine_winner(move1, move2):
    if move1 == move2:
        return "It's a Tie!"
    elif (move1 == "rock" and move2 == "scissors") or \
         (move1 == "scissors" and move2 == "paper") or \
         (move1 == "paper" and move2 == "rock"):
        return "Player 1 Wins!"
    else:
        return "Player 2 Wins!"

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    detected_moves = []  # List to store detected moves

    # Coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values

            # Confidence
            confidence = math.ceil((box.conf[0] * 100)) / 100
            if confidence < 0.70:
                continue

            # Put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Class name
            cls = int(box.cls[0])
            detected_moves.append(classNames[cls])  # Store the detected move

            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    if len(detected_moves) >= 2:  # Ensure we have moves from both players
        move1 = detected_moves[0]
        move2 = detected_moves[1]
        winner = determine_winner(move1, move2)

        # Display the winner on the screen
        cv2.putText(img, winner, (50, 50), font, 1.5, (0, 255, 0), 3)
        print("Detected Moves:", move1, move2)
        print("Winner:", winner)

    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
