cmake_minimum_required(VERSION 3.5)

find_package(Caffe)
IF(Caffe_FOUND)
    MESSAGE(STATUS "caffe found")
ENDIF(Caffe_FOUND)

find_package(OpenCV REQUIRED)
IF(OpenCV_FOUND)
    MESSAGE(STATUS "opencv found")
ENDIF(OpenCV_FOUND)


include_directories("${PROJECT_SOURCE_DIR}/include/") #当前工程下include文件夹 
include_directories(${OpenCV_INCLUDE_DIRS} ${Caffe_INCLUDE_DIRS})

AUX_SOURCE_DIRECTORY(. DIR_SRCS)

add_executable(classifier ${DIR_SRCS})

target_link_libraries(classifier
    ${OpenCV_LIBRARIES}   
    ${Caffe_LIBRARIES}
)

