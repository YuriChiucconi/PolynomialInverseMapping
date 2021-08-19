### test/CMakeList.txt

INCLUDE_DIRECTORIES(
	include
	${CMAKE_SOURCE_DIR}/class/include
	${CMAKE_SOURCE_DIR}/test/include
	)

ADD_EXECUTABLE(test source/test.cc source/main.cc)
DEAL_II_SETUP_TARGET(test)

TARGET_LINK_LIBRARIES(test class)
