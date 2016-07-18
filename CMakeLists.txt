cmake_minimum_required (VERSION 2.8.11)

project(caffefication)

# ------------------------------------------------------------------------------
# C++11 settings
# ------------------------------------------------------------------------------
if(CMAKE_COMPILER_IS_GNUCXX)
    if(NOT CMAKE_CXX_FLAGS MATCHES "-std=c\\+\\+11")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    endif()
endif()


set(CMAKE_CXX_EXTENSIONS 0)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED 1)

add_definitions(-DGLOG_NO_ABBREVIATED_SEVERITIES -DNO_STRICT)

# ------------------------------------------------------------------------------
# Caffe
# ------------------------------------------------------------------------------
find_package( OpenCV REQUIRED )

set(Caffe_INCLUDE_DIRS "caffe/src;caffe/include;caffe/bin/include;/usr/include/opencv;/usr/include/atlas;")
set(Caffe_LIBRARIES "/home/hjimce/workspace/caffecpp/caffe/build/lib/libcaffe.so;/usr/lib/x86_64-linux-gnu/libglog.so")

include_directories(${Caffe_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})


find_package(Boost REQUIRED COMPONENTS system)
include_directories(${Boost_INCLUDE_DIRS})



#find_package(glog REQUIRED)#搜索Protobuf库





message(STATUS "Caffe_DEFINITIONS")
message(STATUS ${Caffe_DEFINITIONS})

message(STATUS "Caffe_INCLUDE_DIRS")
message(STATUS ${Caffe_INCLUDE_DIRS})

message(STATUS "Caffe_LIBRARIES")
message(STATUS ${Caffe_LIBRARIES})


set(SOURCES "testcpp.cpp" "directory.cpp" "directory.h")
ADD_EXECUTABLE(caffefication ${SOURCES} )
target_link_libraries(caffefication ${Caffe_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES} )