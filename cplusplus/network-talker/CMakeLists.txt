INCLUDE(FindProtobuf)
FIND_PACKAGE(Protobuf REQUIRED)
INCLUDE_DIRECTORIES(${PROTOBUF_INCLUDE_DIR})

PROTOBUF_GENERATE_CPP(PROTO_SRC PROTO_HEADER message.proto)
add_custom_target(generate-protobuf-cpp ALL DEPENDS ${PROTO_SRC} ${PROTO_HEADER})
add_custom_command(
        TARGET generate-protobuf-cpp PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${PROTO_SRC}
                ${CMAKE_CURRENT_SOURCE_DIR}/message.pb.cc
        COMMAND ${CMAKE_COMMAND} -E copy
                ${PROTO_HEADER}
                ${CMAKE_CURRENT_SOURCE_DIR}/message.pb.h)

PROTOBUF_GENERATE_PYTHON(PROTO_PY message.proto)
add_custom_target(generate-protobuf-python ALL DEPENDS ${PROTO_PY})
add_custom_command(
        TARGET generate-protobuf-python PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${PROTO_PY}
                ${CMAKE_SOURCE_DIR}/../python/message_pb2.py)

add_executable(network-talker network-talker.cpp ${PROTO_HEADER} ${PROTO_SRC})
target_link_libraries(network-talker protobuf)
