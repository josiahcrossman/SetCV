# SetCV

This is a project designed to have a computer play the game of set using a webcam and traditional image processing techniques. 

## When using this code:

Choose the file path for your image, and then use it in the game variable at the bottom of the SetFinder file. By default the code will use the sample image. 
Set cards should be aligned in the same way as the example image. Try to keep the camera as straight as possible when taking the photo for best results. The code will then show images of all possible sets

### There are two parameters that need to be tuned when using this project with different cameras/environments:
1. The HSV thresholds for color detection will likely need to be tuned, depending on the warmth of the surrounding light. 
2. The size filter for cards in get_cards(image) will also need to be tuned, depening on how far above the card the image is taken and the resolution of the camera.


## Example input and output:
![WIN_20240811_16_42_11_Pro](https://github.com/user-attachments/assets/67c05826-f1ab-485d-9210-843ab059c3af)
![testOutput](https://github.com/user-attachments/assets/27334311-3904-4492-bd07-cc1c6f512d8f)
