#include "proto/message.pb.h"

#include <cassert>
#include <iostream>
#include <sys/socket.h>
#include <arpa/inet.h>

class NetworkTalker
{
public:
  NetworkTalker(uint16_t send_port = 5555, uint16_t receive_port = 5556) {
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    sender_addr.sin_family = AF_INET;
    inet_aton("127.0.0.1", &sender_addr.sin_addr);
    sender_addr.sin_port = htons(send_port);
    send_socket = socket(PF_INET, SOCK_DGRAM, 0);

    receiver_addr.sin_family = AF_INET;
    inet_aton("127.0.0.1", &receiver_addr.sin_addr);
    receiver_addr.sin_port = htons(receive_port);
    receiver_socket = socket(AF_INET, SOCK_DGRAM, 0);
    int result = bind(receiver_socket, reinterpret_cast<struct sockaddr*>(&receiver_addr), sizeof(receiver_addr));
    if (result < 0) {
      std::cerr << "NetworkTalker couldn't bind receiver socket" << std::endl;
    }
  }

  std::string process(const std::string& token) {
    send(token);
    return receive();
  }
private:

  void send(const std::string& token) {
    TypoCorrector::ContaminatedToken foo;
    foo.set_content(token.c_str());

    std::string buf;
    foo.SerializeToString(&buf);
    sendto(send_socket, buf.data(), buf.size(), 0, reinterpret_cast<struct sockaddr *>(&sender_addr), sizeof(sender_addr));
  }

  std::string receive() {
    int received_size = static_cast<int>(recv(receiver_socket, receive_buffer, 1024, 0));
    TypoCorrector::CleanedHypo cleaned_hypo;
    cleaned_hypo.ParseFromArray(receive_buffer, received_size);
    return cleaned_hypo.content();
  }

private:
  struct sockaddr_in sender_addr;
  int send_socket;

  struct sockaddr_in receiver_addr;
  int receiver_socket;
  char receive_buffer[1024];
};

int main()
{
  NetworkTalker talker;
  while (true) {
    std::cout << "Input somehing: ";
    std::string contaminated_token;
    std::getline(std::cin, contaminated_token);
    std::cout << talker.process(contaminated_token) << std::endl << std::endl;
  }
}
