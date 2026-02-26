from ultralytics import YOLO

# Load the YOLOv9 model
model = YOLO('./yolov9c.pt')
#
## Run predictions
results = model.predict(fr'C:\Users\Robb Cenan\OneDrive\Documents\CODING PROJECTS\trafficlights\WEBDEV\abcd.jpg')
#
## Display results
results[0].show()