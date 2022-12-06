import numpy as np
import trimesh
from config import config_parser
import cv2
import torch
from models import net
import mediapipe as mp
from torchvision import transforms
import pyrender
from pathlib import Path
import signal
import sys
from utils.utils import backproject_vertices
import matplotlib.pyplot as plt

def visualize_3d_model(vertices):
    # initially, there might not be a face in the frame, so there wont be a most recent prediction. Skip that situation.
    if vertices is None: return

    vertice = vertices[0,:,:].detach().cpu().numpy().squeeze()
    faces = network.flame_layer.faces_tensor
    vertex_colors = np.ones([vertice.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
    tri_mesh = trimesh.Trimesh(vertice, faces,
                               vertex_colors=vertex_colors)

    # put camera at a distance
    rot_mat = np.array([[1.,0.,0.,0.],
                        [0.,1.,0.,0.],
                        [0.,0.,1.,-0.5],
                        [0.,0.,0.,1.]])
    tri_mesh.apply_transform(rot_mat)

    # use pyrender for visualizing mesh output
    mesh = pyrender.Mesh.from_trimesh(tri_mesh)

    # prepare the scene
    scene = pyrender.Scene()
    s = np.sqrt(2) / 2
    camera_pose = np.array([
        [0.0, -s, s, 0.3],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, s, s, 0.35],
        [0.0, 0.0, 0.0, 1.0],
    ])
    light = pyrender.SpotLight(color=np.ones(3), intensity=0.4,innerConeAngle = np.pi / 16.0,outerConeAngle = np.pi / 6.0)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    scene.add(light, pose=camera_pose)
    scene.add(mesh, pose=camera_pose)
    scene.add(camera, pose=camera_pose)

    # render the scene
    r = pyrender.OffscreenRenderer(args.demo_width, args.demo_height)
    color, _ = r.render(scene)
    img_out = np.array(color)

    if visualize:
        if args.use_webcam:
            # flip for selfie view
            cv2.imshow("Mesh Result", cv2.flip(img_out,1))
        else:
            cv2.imshow("Mesh Result", img_out)

    if args.save_mesh_video:
        # save images as a video
        video.write(img_out)
        print(f"Frame {idx} written to video")


def load_model(network, args, device):
    checkpoint = torch.load(args.checkpoint_model_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    network.load_state_dict(state_dict)
    print("-> loaded checkpoint %s (epoch: %d)" % (args.checkpoint_model_path, checkpoint['epoch']))

def ctrlc_handler(sig, frame):
    # while saving webcam video, release the video when closed by ctrl-c
    print("Handling CTRL-C")
    if args.save_mesh_video and args.use_webcam: video.release()
    sys.exit(0)

if __name__ == '__main__':
    # register exit signal
    signal.signal(signal.SIGINT, ctrlc_handler)

    # load the configs
    parser = config_parser()
    args = parser.parse_args()

    # prcoess one image each time
    args.batch_size = 1

    # visualize the process or not
    visualize = args.visualize_demo

    # check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running code on", device)

    # load the model (use the checkpoint model)
    network = net.Encoder(args, device)
    load_model(network, args, device)
    network.eval()
    torch.set_grad_enabled(False)

    # transform the image same as training data
    transform_image = transforms.Compose([transforms.ToTensor(), transforms.Resize([args.height, args.width]),
                                          transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), ])

    # MediaPipe's Face Detection
    mp_face_detection = mp.solutions.face_detection
    # mp_drawing = mp.solutions.drawing_utils

    # Use webcam video or another video file
    if args.use_webcam:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.demo_video_path)

    # initialize the video writer object
    if args.save_mesh_video:
        if not Path("demo/results").exists():
            Path("demo/results").mkdir(parents=True, exist_ok=True)
         
        video = cv2.VideoWriter(args.output_video_path, -1, cap.get(cv2.CAP_PROP_FPS), (args.demo_width, args.demo_height))
    idx = 0

    # initialize vertices as none
    vertices = None

    # initialize the face detection
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=args.detection_threshold)

    while cap.isOpened():
        success, image = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            if args.use_webcam:
                continue
            else:
                cv2.destroyAllWindows()
                if args.save_mesh_video: video.release()
                print("Done!")
                break


        # should make it faster
        image.flags.writeable = False

        # copy RGB order image before normalizations etc.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb = image.copy() 
    
        # get image dimensions
        img_shape = image.shape
        height, width = img_shape[0], img_shape[1]

        #get face detection results
        results = face_detection.process(image) 
        
        #change back to BGR for cv2 visualization
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
        
        # check if there are detections
        if results.detections:
            # if there are multiple faces detected in the video, ignore the frame
            if len(results.detections) > 1:
                print("Warning: Multiple faces detected, ignoring the frame")
            else:
                detection = results.detections[0]

                # get and enlarge the bounding box
                bb = detection.location_data.relative_bounding_box
                x, y = bb.xmin*width, bb.ymin*height
                w, h = bb.width*width, bb.height*height
                x_e, y_e = x-0.15*w, y-0.2*h
                w_e, h_e = 1.2*w, 1.35*h

                # if the bounding box is out of the frame, ignore the frame
                if x_e > 0 and y_e > 0 and x_e+w_e<width and y_e+h_e<height:
                    # get the enlarged face crop
                    face_crop = image_rgb[round(y_e):round(y_e+h_e), round(x_e):round(x_e+w_e)]
                    
                    # original dimensions before transforming
                    orig_height, orig_width = face_crop.shape[1], face_crop.shape[0]

                    # transform the image
                    input_img = transform_image(face_crop)[None,...]

                    # get dimensions after transforming
                    mod_height, mod_width = input_img.shape[2], input_img.shape[3]
                    
                    # calculate dimension transforms
                    tf_x, tf_y = torch.tensor(mod_height/orig_height)[None,...], torch.tensor(mod_width/orig_width)[None,...] 

                    # get the prediction
                    prediction, vertices = network(input_img, (tf_x, tf_y))
    
                else: 
                    print("Warning: All face is not visible, ignoring the frame")

        else:
            # cannot find a face in current frame, ignore the frame
            print("Warning: No faces detected, ignoring the frame")
        
        # if the frame is not ignored, display the current prediction, otherwise display the latest
        visualize_3d_model(vertices)
        idx += 1

        # backproject vertices to the original image to display
        if args.backproject_vertices and (vertices is not None):
            projected_vertices, _ = backproject_vertices(prediction, vertices, (tf_x, tf_y), network.scale)
            # scatter dots on vertex locations
            for v in range(projected_vertices.size(1)):
                center_x = round(float(projected_vertices[0,v,0] +round(x_e)))
                center_y = round(float(projected_vertices[0,v,1] +round(y_e)))
                image = cv2.circle(image, (center_x, center_y), radius=0, color=(0, 0, 255), thickness=4)
    
        # display the prediction or not
        if visualize:
            if args.use_webcam:
                # Flip the image horizontally for a selfie-view display.
                cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
            else:
                cv2.imshow('MediaPipe Face Detection', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()
