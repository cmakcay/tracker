import configargparse

def config_parser():
    parser = configargparse.ArgumentParser()

    # training options
    parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--continue_from_epoch", type=int, default=0, help="continue training from epoch")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers to use during batch generation")
    parser.add_argument("--num_classes", type=int, default=159, help="number of encoder parameters to predict")
    parser.add_argument("--reg_term", type=int, default=0.001, help="regularization term for regularization losses")

    # demo options
    parser.add_argument("--checkpoint_model_path", type=str, default="chkpts/tracker.pt", help="trained model used for demo")
    parser.add_argument("--demo_video_path", type=str, default="demo/vid/video_name.mp4")
    parser.add_argument("--output_video_path", type=str, default="demo/results/video_name.mp4")
    parser.add_argument("--use_webcam", type=bool, default=True)
    parser.add_argument("--save_mesh_video", type=bool, default=False)
    parser.add_argument("--visualize_demo", type=bool, default=True, help="visualize the results of demo simultaneously")
    parser.add_argument("--backproject_vertices", type=bool, default=True, help="backproject vertices back to image or not")
    parser.add_argument("--detection_threshold", type=float, default=0.7, help="face detection threshold of MediaPipe")
    parser.add_argument("--demo_width", type=int, default=1000, help="face detection threshold of MediaPipe")
    parser.add_argument("--demo_height", type=int, default=1000, help="face detection threshold of MediaPipe")

    #NoW options
    parser.add_argument("--NoW_image_path", type=str, default=".../NoW/imagepathsvalidation.txt")
    parser.add_argument("--NoW_dataset_path", type=str, default=".../NoW_Dataset/final_release_version")
    parser.add_argument("--NoW_output_path", type=str, default=".../NoW_Dataset/predicted_meshes")

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='vgg_face2')
    parser.add_argument("--csv_file_train", type=str, default="data/landmarks/train_list.csv")
    parser.add_argument("--csv_file_val", type=str, default="data/landmarks/val_list.csv")
    parser.add_argument("--root_dir", type=str, default=".../VGG-Face2")
    parser.add_argument("--width", type=int, default=224, help="width")
    parser.add_argument("--height", type=int, default=224, help="height")

    # logging/saving options
    parser.add_argument("--checkpoints_dir", type=str, help='specify the directory to save the model', default='chkpts/')
    parser.add_argument("--save_every", type=str, default=10,  help='save model every 10 epochs')
    parser.add_argument("--val_every", type=str, default=1, help='validate model every epoch')
    parser.add_argument("--log_every", type=str, default=1000, help='save logs every 1000 batches')

    # FLAME options
    parser.add_argument('--flame_lmk_embedding_path', type=str, default='data/flame_files/landmark_embedding.npy', help='flame model path')
    parser.add_argument('--flame_model_path', type=str, default='data/flame_files/generic_model.pkl', help='flame model path')
    parser.add_argument('--shape_params', type=int, default=100, help='the number of shape parameters')
    parser.add_argument('--expression_params', type=int, default=50, help='the number of expression parameters')
    parser.add_argument('--pose_params', type=int, default=6, help='the number of pose parameters')

    return parser

