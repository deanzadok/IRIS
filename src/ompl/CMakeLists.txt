add_library(ompl crisp_state_space.cc crisp_directed_control_sampler.cc control_rrg.cc control_rrt.cc control_rrtstar.cc)
target_link_libraries(ompl crisp_robot ${OMPL_LIBRARIES})



set (VISILIBITY_SRC "/home/deanz/Documents/Github/VisiLibity1/src")
include_directories(${VISILIBITY_SRC})

set (VISILIBITY_SCPP "${VISILIBITY_SRC}/visilibity.cpp")
set (VISILIBITY_SHPP "${VISILIBITY_SRC}/visilibity.hpp")

add_library(rrg rrg.cc ${VISILIBITY_SCPP} ${VISILIBITY_SHPP} )
target_link_libraries(rrg ${OMPL_LIBRARIES})
target_link_libraries(rrg ${OpenCV_LIBS})
target_link_libraries(rrg tensorflow)

if (USE_C++17)
    add_library(drone drone_state_space.cc)
    target_link_libraries(drone drone_robot ${OMPL_LIBRARIES})
endif()
