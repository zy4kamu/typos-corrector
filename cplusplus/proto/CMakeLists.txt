INCLUDE(FindProtobuf)
FIND_PACKAGE(Protobuf REQUIRED)
INCLUDE_DIRECTORIES(${PROTOBUF_INCLUDE_DIR})

PROTOBUF_GENERATE_CPP(PROTO_SRC PROTO_HEADER message.proto)
ADD_LIBRARY(proto ${PROTO_HEADER} ${PROTO_SRC})

PROTOBUF_GENERATE_PYTHON(PROTO_PY message.proto)
add_custom_target(proto-py ALL DEPENDS ${PROTO_PY})
add_custom_command(
        TARGET proto-py POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${PROTO_PY}
                ${CMAKE_SOURCE_DIR}/../python/message_pb2.py)

