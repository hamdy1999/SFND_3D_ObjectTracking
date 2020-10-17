# SFND 3D Object Tracking

Welcome to the final project of the camera course. By completing all the lessons, you now have a solid understanding of keypoint detectors, descriptors, and methods to match them between successive images. Also, you know how to detect objects in an image using the YOLO deep-learning framework. And finally, you know how to associate regions in a camera image with Lidar points in 3D space. Let's take a look at our program schematic to see what we already have accomplished and what's still missing.

<img src="images/course_code_structure.png" width="779" height="414" />

In this final project, you will implement the missing parts in the schematic. To do this, you will complete four major tasks: 
1. First, you will develop a way to match 3D objects over time by using keypoint correspondences. 
2. Second, you will compute the TTC based on Lidar measurements. 
3. You will then proceed to do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 
4. And lastly, you will conduct various tests with the framework. Your goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor. In the last course of this Nanodegree, you will learn about the Kalman filter, which is a great way to combine the two independent TTC measurements into an improved version which is much more reliable than a single sensor alone can be. But before we think about such things, let us focus on your final project in the camera course. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* Git LFS
  * Weight files are handled using [LFS](https://git-lfs.github.com/)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking`.

## Rubric

### 1. Match 3D Objects

**Criteria:**
Implement the method "matchBoundingBoxes", which takes as input both the previous and the current data frames and provides as output the ids of the matched regions of interest (i.e. the boxID property). Matches must be the ones with the highest number of keypoint correspondences.

**Solution:**
Construct `prev_curr_box_score` which records the frequency / score of box-box pair. Iterate matched key points. If the matched points belong to the bounding box pair, then the corresponding box-box pair score increse by one. Last, for each quiry box, select the train box id which has the biggest score.

```c++
void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{

    // ...
    int prev_box_size = prevFrame.boundingBoxes.size();
    int curr_box_size = currFrame.boundingBoxes.size();
    //int prev_curr_box_score[prev_box_size][curr_box_size] = {};
    vector<vector<int>> matchScore(prev_box_size, vector<int>(curr_box_size, 0));
    // iterate pnt matchs, cnt box-box match score
    for (auto it = matches.begin(); it != matches.end() - 1; it++)
    {
        // prev pnt
        cv::KeyPoint prev_key_pnt = prevFrame.keypoints[it->queryIdx];
        cv::Point prev_pnt = cv::Point(prev_key_pnt.pt.x, prev_key_pnt.pt.y);

        // curr pnt
        cv::KeyPoint curr_key_pnt = currFrame.keypoints[it->trainIdx];
        cv::Point curr_pnt = cv::Point(curr_key_pnt.pt.x, curr_key_pnt.pt.y);

        // get corresponding box with the point
        std::vector<int> prev_box_id_list, curr_box_id_list;

        for (int i = 0; i < prev_box_size; ++i)
            if (prevFrame.boundingBoxes[i].roi.contains(prev_pnt))
                prev_box_id_list.push_back(i);

        for (int j = 0; j < curr_box_size; ++j)
            if (currFrame.boundingBoxes[j].roi.contains(curr_pnt))
                curr_box_id_list.push_back(j);

        // add cnt to prev_curr_box_score
        for (int i : prev_box_id_list)
            for (int j : curr_box_id_list)
                matchScore[i][j] += 1;

    } // END OF THE PNT MATCH

    // for each box in prevFrame, find the box with highest score in currFrame
    for (int i = 0; i < prev_box_size; ++i)
    {
        int max_score = 0;
        int best_idx = 0;

        for (int j = 0; j < curr_box_size; ++j)
            if (matchScore[i][j] > max_score)
            {
                max_score = matchScore[i][j];
                best_idx = j;
            }


        bbBestMatches[i] = best_idx;
    }
}
```

### 2. Compute Lidar-based TTC

**Criteria:**
Compute the time-to-collision in second for all matched 3D objects using only Lidar measurements from the matched bounding boxes between current and previous frame.

**Solution:**
In order to deal with the outlier points, only select lidar points within ego lane, then use 10% quantile to get stable distance estimation. Finally, use distance estimation to calculate ttc.

ttc = d1 * t / (d0 - d1)

```c++
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // ...
    double dT = 1.0 / frameRate;

    vector<double> xPrev, xCurr;
    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end() - 1; ++it)
        xPrev.push_back(it->x);
    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end() - 1; ++it)
        xCurr.push_back(it->x);

    sort(xPrev.begin(), xPrev.end());
    sort(xCurr.begin(), xCurr.end());

    double minXPrev = 1e9, minXCurr = 1e9;

    //the point is robust if it's surrounded by more than 5 points that share a close value
    if (xPrev.size() > 10)
        for (auto it = xPrev.begin(); it != xPrev.end(); ++it)
        {
            for (int i = 1; i <= 5; i++) //assert that at least the 5 points are close togegher
            {
                if (*(it + i) - *it > 0.1) // if not
                {
                    it = it + i; // take the big point
                    i = 0; // re-assert
                }
                minXPrev = *it;
            }

            break;// the condition is valid
        }
    else //if the No. points is small
        minXPrev = xPrev[(int)(xPrev.size() / 2)]; //the middle value

    //same for xCurr
    if (xPrev.size() > 10)
        for (auto it = xCurr.begin(); it != xCurr.end(); ++it)
    {
        for (int i = 0; i <= 5; i++)
        {
            if (*(it + i) - *it > 0.1)
            {
                it = it + i;
                i = 0;
            }
            minXCurr = *it;
        }

        break;
    }
    else
        minXCurr = xCurr[(int)(xCurr.size()/2)]; //the middle value

    TTC = minXCurr * dT / (minXPrev - minXCurr);

}
```

### 3. Associate Keypoint Correspondences with Bounding Boxes

**Criteria:**
Prepare the TTC computation based on camera measurements by associating keypoint correspondences to the bounding boxes which enclose them. All matches which satisfy this condition must be added to a vector in the respective bounding box.

**Solution:**
First filter key point matches according to the distance to mean match distance. Then associate a given bounding box with the keypoints it contains and the corresponding matches.

```c++
// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // calculate mean point match distance in this bbox
    double distance_mean = 0.0;
    double size = 0.0;
    for (auto it = kptMatches.begin(); it != kptMatches.end(); ++it)
    {
        cv::KeyPoint curr_pnt = kptsCurr[it->trainIdx];
        cv::KeyPoint prev_pnt = kptsPrev[it->queryIdx];

        if (boundingBox.roi.contains(curr_pnt.pt))
        {
            distance_mean += cv::norm(curr_pnt.pt - prev_pnt.pt);
            size += 1;
        }
    }
    distance_mean = distance_mean / size;

    // filter point match based on point match distance
    for (auto it = kptMatches.begin(); it != kptMatches.end(); ++it)
    {
        cv::KeyPoint curr_pnt = kptsCurr[it->trainIdx];
        cv::KeyPoint prev_pnt = kptsPrev[it->queryIdx];

        if (boundingBox.roi.contains(curr_pnt.pt))
        {
            double curr_dist = cv::norm(curr_pnt.pt - prev_pnt.pt);

            if (curr_dist < distance_mean * 1.3)
            {
                boundingBox.keypoints.push_back(curr_pnt);
                boundingBox.kptMatches.push_back(*it);
            }
        }
    }
}
```

### 4. Compute Camera-based TTC

**Criteria:**
Compute the time-to-collision in second for all matched 3D objects using only keypoint correspondences from the matched bounding boxes between current and previous frame.

**Solution:**
Calculate the distance change from previous key point pairs to current matched key point pairs. The use the formula below to get the camera based ttc.

ttc = -t / (1-d1/d0)

```c++
// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    // STUDENT TASK (replacement for meanDistRatio)
    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);
    // EOF STUDENT TASK
}

```

## FP.5 - Performance Evaluation 1 - LiDAR-based TTC

The table below shows the TTC estimate at a particular frame index. Recall that we are using LiDAR points that are bounded within object bounding boxes created by the YOLO Object Detection algorithm, so this is independent of any feature detectors and descriptors used. We will also show the minimum `x` coordinate in the point cloud at each frame to help with the analysis

| Image Index | TTC LiDAR (in seconds) | Min `x` coordinate (in m) |
|:-----------:|:----------------------:|---------------------------|
|      1      |        12.9722         | 7.913                     |
|      2      |         12.264         | 7.849                     |
|      3      |        13.9161         | 7.793                     |
|      4      |        7.11572         | 7.685                     |
|      5      |        16.2511         | 7.638                     |
|      6      |        12.4213         | 7.577                     |
|      7      |        34.3404         | 7.555                     |
|      8      |        9.34376         | 7.475                     |
|      9      |        18.1318         | 7.434                     |
|     10      |        18.0318         | 7.393                     |
|     11      |        14.9877         | 7.344                     |
|     12      |          10.1          | 7.272                     |
|     13      |        9.22307         | 7.194                     |
|     14      |        10.9678         | 7.129                     |
|     15      |        8.09422         | 7.042                     |
|     16      |        8.81392         | 6.963                     |
|     17      |        10.2926         | 6.896                     |
|     18      |        8.30978         | 6.814                     |

The first three frames seem plausible but when we get to frame 4, it drops right down to 7.117 seconds. The reason why is because the minimum distance dropped from 7.79 m from frame 3 down to 7.68 m to frame 4\. For the first three frames, the distance to the ego vehicle seems plausible as it looks like we're gradually slowing down. However, at frame 4 there is a sudden jump in closeness meaning that the TTC thus decreases more quickly. The first three frames have a difference of 0.06 m between successive frames, but for frame 4, we have a 0.11 m difference. This is roughly double the distance resulting in the TTC decreasing by half in proportion.

To see what's going on, the two images below show the bird's eye view of the preceding vehicle's point cloud from frames 3 and 4.

![Frame 3 Preceding Vehicle Point Cloud](images/results/frame3_lidar.png)

![Frame 4 Preceding Vehicle Point Cloud](images/results/frame4_lidar.png)

We can see that shape of the point cloud of the back of the preceding vehicle is quite similar between the two frames. However, the cluster of points is shifted by the distances mentioned, thus reporting the incorrect TTC. It is not so much the point cloud but the constant velocity model breaks with sudden changes in distance. If we used a constant acceleration model this would report a more accurate TTC. Under the constant velocity model, this certainly goes to show that using LiDAR on its own is not a reliable source of information to calculate the TTC. Another instance is from frames 6 to 7 where the distances shorten from 7.58 m to 7.55 m, but the TTC increases to 34.34 seconds, but clearly we can visually see that the appearance of the preceding vehicle has not changed much between the frames so this is obviously a false reading. The two images below show the bird's eye view of the preceding vehicle's point cloud from frames 6 and 7.

![Frame 6 Preceding Vehicle Point Cloud](images/results/frame6_lidar.png)

![Frame 7 Preceding Vehicle Point Cloud](images/results/frame7_lidar.png)

As we can clearly see, the point clouds are well formed but due to the constant velocity model, a short displacement between frames breaks down this assumption quickly.


### 6. Performance Evaluation 2

**Criteria:**
Run several detector / descriptor combinations and look at the differences in TTC estimation. Find out which methods perform best and also include several examples where camera-based TTC estimation is way off. As with Lidar, describe your observations again and also look into potential reasons.

**Solution:**

I used some combinations in the comparison for the same first frame and here's the result.

|Img. No. | Detector + Descriptor | TTC Lidar | TTC Camera |
|:---:|:---:|:----:|:-----:|
| 1 | SHITOMASI + BRISK | 12.9722 | 15.3278 |
| 1 | SHITOMASI + BRIEF | 12.9722 | 14.6756 |
| 1 | SHITOMASI + ORB | 12.9722 | 13.8743 |
| 1 | SHITOMASI + FREAK | 12.9722 | 15.9994 |
| 1 | SHITOMASI + SIFT | 12.9722 | 14.0134 |
| 1 | HARRIS + SIFT | 12.9722 | 10.9082 |
| 1 | ORB + SIFT | 12.9722 | 15.8832 |
| 1 | FAST + BRISK | 12.9722 | 12.3379 |
| 1 | FAST + BRIEF | 12.9722 | 10.8026 |
| 1 | FAST + ORB | 12.9722 | 11.0105 |
| 1 | FAST + FREAK | 12.9722 | 12.3974 |
| 1 | FAST + SIFT | 12.9722 | 12.6908 |
| 1 | SIFT + SIFT | 12.9722 | 11.6339 |
| 1 | SIFT + BRISK | 12.9722 | 13.5175 |
| 1 | SIFT + BRIEF | 12.9722 | 11.9962 |
| 1 | SIFT + FREAK | 12.9722 | 11.578 |
| 1 | BRISK + BRISK | 12.9722 | 13.9354 |
| 1 | BRISK + FREAK | 12.9722 | 14.4862 |
| 1 | BRISK + BRIEF | 12.9722 | 13.8215 |
| 1 | BRISK + ORB | 12.9722 | 13.8211 |
| 1 | BRISK + SIFT | 12.9722 | 15.8393 |
| 1 | AKAZE + AKAZE | 12.9722 | 12.5835 |
| 1 | AKAZE + BRIEF | 12.9722 | 13.3774 |
| 1 | AKAZE + ORB | 12.9722 | 12.8277 |
| 1 | AKAZE + FREAK | 12.9722 | 12.773 |
| 1 | AKAZE + SIFT | 12.9722 | 12.6613 |
| 1 | AKAZE + BRISK| 12.9722 | 12.2899 |


In the mid-term project, the top 3 detector/descriptor has been seletected in terms of their performance on accuracy and speed. So here, we use them one by one for Camera TTC estimate.

|Sr. No. | Detector + Descriptor | notes |
|:---:|:---:|:---:|
|1 | FAST + BRIEF | Best Efficiency |
|2 | AKAZE + BRIEF | Moderate |
|3 | BRISK + BRIEF | Best Accuracy |

The TTC estimation results are shown in the table below.

|Sr. No. | lidar | FAST + BRIEF | AKAZE + BRIEF | BRISK + BRIEF |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 12.972 | 10.802 | 13.377 | 13.8215 |
| 2 | 12.264 | 11.006 | 15.197 | 18.0906 |
| 3 | 13.916 | 14.155 | 13.141 | 12.4078 |
| 4 | 7.1157 | 14.388 | 14.595 | 21.8235 |
| 5 | 16.251 | 19.951 | 15.605 | 25.9018 |
| 6 | 12.421 | 13.293 | 13.640 | 19.1942 |
| 7 | 34.340 | 12.218 | 15.739 | 17.9009 |
| 8 | 9.3437 | 12.759 | 14.459 | 18.3884 |
| 9 | 18.131 | 12.6 | 14.113 |14.9617 |
| 10 | 18.031 | 13.463 | 11.896 | 12.0464 |
| 11 | 14.987 | 13.671 | 13.025 | 11.5343 |
| 12 | 10.1 | 10.9087 | 12.168 | 14.1488 |
| 13 | 9.2230 | 12.370 | 10.248 | 12.8926 |
| 14 | 10.967 | 11.243 | 10.126 | 10.3194 |
| 15 | 8.0942 | 11.874 | 9.1507 | 11.1036 |
| 16 | 8.8139 | 11.839 | 10.145 | 13.047 |
| 17 | 10.292 | 7.9201 | 9.5386 | 11.1959 |
| 18 | 8.3097 | 11.554 | 8.8438 | 9.57089 |


There are matched points in the ground or in other cars which violates the assumption that each matched point has same distance to the ego car. The camera ttc is much unstable compared with lidar ttc.

 <img src="images/results/camera.png"  height="120">
 <img src="images/results/camera2.png"  height="120">