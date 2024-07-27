#include <opencv2/opencv.hpp>
#include <iostream>

cv::Point2f updatePositionAlongRectangle(cv::Point2f currentPosition, cv::Point2f& direction, const cv::Rect& rect) {
    currentPosition += direction;

    if (currentPosition.x <= rect.x || currentPosition.x >= rect.x + rect.width) {
        direction.x = -direction.x;
    }
    if (currentPosition.y <= rect.y || currentPosition.y >= rect.y + rect.height) {
        direction.y = -direction.y;
    }

    return currentPosition;
}

int main() {

    cv::namedWindow("Kalman Filter Tracking",0);
    cv::KalmanFilter KF(4, 2, 0);

    KF.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, 1, 0,
                           0, 1, 0, 1,
                           0, 0, 1, 0,
                           0, 0, 0, 1);

    KF.measurementMatrix = (cv::Mat_<float>(2, 4) << 1, 0, 0, 0,
                            0, 1, 0, 0);

    setIdentity(KF.processNoiseCov, cv::Scalar(1e-4));

    setIdentity(KF.measurementNoiseCov, cv::Scalar(1e-1));

    setIdentity(KF.errorCovPost, cv::Scalar(1));

    KF.statePost = (cv::Mat_<float>(4, 1) << 0, 0, 0, 0);

    int width = 640;
    int height = 480;
    cv::Mat frame(height, width, CV_8UC3, cv::Scalar(0, 0, 0));

    cv::Rect rectangle(100, 100, 440, 280);

    cv::Point2f truePosition(rectangle.x + rectangle.width / 2, rectangle.y + rectangle.height / 2);
    cv::Point2f direction(5, 5);

    cv::Point2f predictedPosition;

    while (true) {
        frame.setTo(cv::Scalar(0, 0, 0));

        truePosition = updatePositionAlongRectangle(truePosition, direction, rectangle);

        cv::Mat prediction = KF.predict();
        predictedPosition = cv::Point2f(prediction.at<float>(0), prediction.at<float>(1));

        cv::Mat measurement = (cv::Mat_<float>(2, 1) << truePosition.x, truePosition.y);
        KF.correct(measurement);

        cv::Mat future_state = KF.statePost.clone();
        for (int i = 0; i < 10; ++i) { // predicting 10 steps ahead
            future_state = KF.transitionMatrix * future_state;
        }
        cv::Point futurePredictPt(future_state.at<float>(0), future_state.at<float>(1));

        cv::rectangle(frame, rectangle, cv::Scalar(255, 0, 0), 2); // Rectangle in blue
        cv::circle(frame, truePosition, 10, cv::Scalar(255, 255, 255), -1); // True position in white
        cv::circle(frame, predictedPosition, 10, cv::Scalar(0, 255, 0), -1); // Predicted position in green
        cv::circle(frame, futurePredictPt, 10, cv::Scalar(0, 0, 255), -1); // Predicted position in green

        cv::imshow("Kalman Filter Tracking", frame);

        cv::waitKey(0);
    }

    return 0;
}
