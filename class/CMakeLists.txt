### class/CMakeList.txt

INCLUDE_DIRECTORIES(include)

ADD_LIBRARY(class
	source/class.cc
	)

DEAL_II_SETUP_TARGET(class)
