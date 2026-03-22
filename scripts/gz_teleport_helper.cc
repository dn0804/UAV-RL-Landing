#include <iostream>
#include <string>
#include <gz/transport/Node.hh>
#include <gz/msgs/pose.pb.h>
#include <gz/msgs/boolean.pb.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: gz_teleport_helper <world_name>" << std::endl;
        return 1;
    }
    std::string world = argv[1];
    std::string service = "/world/" + world + "/set_pose";

    gz::transport::Node node;
    gz::msgs::Pose req;
    gz::msgs::Boolean rep;
    bool result;
    unsigned int timeout = 100;

    // Signal ready
    std::cout << "READY" << std::endl;

    // Read "name x y z qw qx qy qz" from stdin in a loop
    std::string name;
    double x, y, z, qw, qx, qy, qz;

    while (std::cin >> name >> x >> y >> z >> qw >> qx >> qy >> qz) {
        req.set_name(name);
        req.mutable_position()->set_x(x);
        req.mutable_position()->set_y(y);
        req.mutable_position()->set_z(z);
        req.mutable_orientation()->set_w(qw);
        req.mutable_orientation()->set_x(qx);
        req.mutable_orientation()->set_y(qy);
        req.mutable_orientation()->set_z(qz);

        node.Request(service, req, timeout, rep, result);

        std::cout << (result ? "OK" : "FAIL") << std::endl;
    }

    return 0;
}
