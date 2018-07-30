import socket
import message_pb2

LOCALHOST = "127.0.0.1"
MESSAGE_SIZE = 1024

class ProtobufTalker(object):
    def __init__(self, receive_port=5555, send_port=5556):
        self.__receive_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.__receive_sock.bind((LOCALHOST, receive_port))
        self.__send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def communicate(self):
        while True:
            received_bytes, _ = self.__receive_sock.recvfrom(MESSAGE_SIZE)
            contaminated_token = message_pb2.ContaminatedToken()
            contaminated_token.ParseFromString(received_bytes)

            cleaned_hypo = message_pb2.CleanedHypo()
            cleaned_hypo.content = self.__process_contaminated_token(contaminated_token.content)
            bytes_to_send = cleaned_hypo.SerializeToString()
            self.__send_sock.sendto(bytes_to_send, ("127.0.0.1", 5556))

    def __process_contaminated_token(self, token):
        return token + "1234"

if __name__ == "__main__":
    talker = ProtobufTalker()
    talker.communicate()
