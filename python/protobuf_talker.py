import socket
import message_pb2

LOCALHOST = "127.0.0.1"
MESSAGE_SIZE = 1024

class ProtobufTalker(object):
    def __init__(self, receive_port=5555):
        self.__receive_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.__receive_sock.bind((LOCALHOST, receive_port))
        self.__send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def receive(self):
        received_bytes, address = self.__receive_sock.recvfrom(MESSAGE_SIZE)
        contaminated_token = message_pb2.ContaminatedToken()
        contaminated_token.ParseFromString(received_bytes)
        return address, str(contaminated_token.content)

    def send(self, address, message):
        cleaned_hypo = message_pb2.CleanedHypo()
        cleaned_hypo.content = message
        bytes_to_send = cleaned_hypo.SerializeToString()
        self.__send_sock.sendto(bytes_to_send, address)

if __name__ == "__main__":
    talker = ProtobufTalker()
    while True:
        received = talker.receive()
        talker.send(received + "1234")
