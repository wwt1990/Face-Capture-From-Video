import cv2
import argparse
import multiprocessing

def take_snapshot_and_save(input_file, output_dir, start, end):
    vidcap = cv2.VideoCapture(input_file)
    total_count = int(vidcap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    vidcap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, start)
    while True:
        success, frame = vidcap.read()
        if not success:
            break
        frameId = int(vidcap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
        if frameId == total_count - 180*25:
            break
        if end != None and frameId >= end:
        	break
        if frameId % 4 == 0:
            cv2.imwrite("{0}/frame_id_{1}.jpg".format(output_dir, frameId), frame)
    vidcap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs',
                        type=int,
                        help="Number of slaves to do the job.",
                        default=1)
    parser.add_argument('--input_file',
                        type=str,
                        help="Path to the video.",
                        required=True)
    parser.add_argument('--output_dir',
                        type=str,
                        help="Path to output full captures of an episode.",
                        required=True)

    args = parser.parse_args()
    num_jobs = args.jobs
    print "num of jobes: {0}".format(num_jobs)

    input_file = args.input_file
    print "input file: {0}".format(input_file)

    output_dir = args.output_dir
    print "output dir: {0}".format(output_dir)

    vidcap = cv2.VideoCapture(input_file)
    total_num_frames = vidcap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    vidcap.release()

    print "Total frames: {0}".format(total_num_frames)

    each_worker_num_frames = total_num_frames / num_jobs

    jobs = []
    
    take_snapshot_and_save(input_file, output_dir, 3220, None)
    # for i in range(num_jobs):
    #     start = int(i * each_worker_num_frames)
    #     end = int(i * each_worker_num_frames + each_worker_num_frames) if i != num_jobs - 1 else None
    #     p = multiprocessing.Process(
    #                 target=take_snapshot_and_save,
    #                 args=(vidcap, input_file, output_dir, start, end))
    #     jobs.append(p)
    #     p.start()

    # for p in jobs:
    #     p.join()
    
    cv2.destroyAllWindows()
