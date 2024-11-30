import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the .npz file
# data = np.load("/home/billyqu/CSCE566_final_project/Problem_1_Diabetic_Retinopathy_Detection_using_Color_Fundus_Photos/ODIR_Data/val/0_right.npz")
data = np.load("/home/billyqu/CSCE566_final_project/Problem_1_Diabetic_Retinopathy_Detection_using_Color_Fundus_Photos/ODIR_Data/train/2_right.npz")

# Extract the image array from the 'slo_fundus' key
image_array = data["slo_fundus"]

# Display the image to verify
plt.imshow(image_array)
plt.axis("off")
plt.title("Reconstructed Image")
plt.show()


print("Race:", data["race"])
print("Male:", data["male"])
print("Hispanic:", data["hispanic"])
print("Marital Status:", data["maritalstatus"])
print("Language:", data["language"])
print("DR Class:", data["dr_class"])
print("DR Subtype:", data["dr_subtype"])

# Save the image as a PNG file
image = Image.fromarray(image_array)
# image.save("0_right.png")

print("Image saved as 1_left.png")