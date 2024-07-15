import sys
import dlib

#if len(sys.argv) < 3:
    #print(
        #"Call this program like this:\n"
        #"   ./cnn_face_detector.py mmod_human_face_detector.dat ../examples/faces/*.jpg\n"
        #"You can get the mmod_human_face_detector.dat file from:\n"
        #"    http://dlib.net/files/mmod_human_face_detector.dat.bz2")
    #exit()

face_rec_model_path = "mmod_human_face_detector.dat"
shape_predict_model = "dlib_face_recognition_resnet_model_v1.dat"
predictor_path = "shape_predictor_5_face_landmarks.dat"
f = "test-image.jpg"

cnn_face_detector = dlib.cnn_face_detection_model_v1(face_rec_model_path)
facerec = dlib.face_recognition_model_v1(shape_predict_model)

win = dlib.image_window()

#for f in image_file:
    #print("Processing file: {}".format(f))
img = dlib.load_rgb_image(f)
# The 1 in the second argument indicates that we should upsample the image
# 1 time.  This will make everything bigger and allow us to detect more
# faces.
dets = cnn_face_detector(img, 1)
'''
This detector returns a mmod_rectangles object. This object contains a list of mmod_rectangle objects.
These objects can be accessed by simply iterating over the mmod_rectangles object
The mmod_rectangle object has two member variables, a dlib.rectangle object, and a confidence score.

It is also possible to pass a list of images to the detector.
    - like this: dets = cnn_face_detector([image list], upsample_num, batch_size = 128)
In this case it will return a mmod_rectangless object.
This object behaves just like a list of lists and can be iterated over.
'''
print("Number of faces detected: {}".format(len(dets)))
for i, d in enumerate(dets):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(
        i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))

rects = dlib.rectangles()
rects.extend([d.rect for d in dets])

win.clear_overlay()
win.set_image(img)
win.add_overlay(rects)


sp = dlib.shape_predictor(predictor_path) 
shape = sp(img, d)
print("Computing descriptor on aligned image ..")
        
        # Let's generate the aligned image using get_face_chip
face_chip = dlib.get_face_chip(img, shape)       


# Now we simply pass this chip (aligned image) to the api
face_descriptor_from_prealigned_image = facerec.compute_face_descriptor(face_chip)                
print(face_descriptor_from_prealigned_image) 
dlib.hit_enter_to_continue()