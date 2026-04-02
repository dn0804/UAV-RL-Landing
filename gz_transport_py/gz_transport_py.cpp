/**
 * gz_transport_py — Minimal Python bindings for gz-transport12.
 *
 * Wraps exactly the functionality DroneEnv needs:
 *   Subscribe:  Odometry, Image
 *   Publish:    Twist, Boolean, Empty
 *
 * Callbacks run on gz-transport's internal threads.  We acquire the
 * Python GIL before invoking the Python callback so it can safely
 * set threading.Events, store data, etc.
 *
 * Build:
 *   mkdir build && cd build
 *   cmake .. && make -j$(nproc)
 *   cp gz_transport_py*.so /path/to/your/project/
 */

#include <atomic>
#include <string>
#include <unordered_map>

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

#include <gz/transport/Node.hh>
#include <gz/msgs/odometry.pb.h>
#include <gz/msgs/image.pb.h>
#include <gz/msgs/twist.pb.h>
#include <gz/msgs/boolean.pb.h>
#include <gz/msgs/empty.pb.h>

namespace py = pybind11;


class GzNode {
public:
    GzNode() = default;

    ~GzNode() {
        active_.store(false);
    }

    // ────────────────────────────────────────────────────────
    //  Subscriptions
    // ────────────────────────────────────────────────────────

    /**
     * Subscribe to a gz::msgs::Odometry topic.
     *
     * Callback signature (all doubles):
     *   cb(px, py, pz, qw, qx, qy, qz, vx, vy, vz)
     *
     * Position  = pose.position      (world frame)
     * Orient.   = pose.orientation   (quaternion, w-first)
     * Velocity  = twist.linear       (world frame)
     */
    bool subscribe_odom(const std::string &topic, py::function cb) {
        odom_cb_ = std::move(cb);
        return node_.Subscribe<gz::msgs::Odometry>(
            topic,
            std::function<void(const gz::msgs::Odometry &)>(
                [this](const gz::msgs::Odometry &msg) { _on_odom(msg); }
            ));
    }

    /**
     * Subscribe to a gz::msgs::Image topic.
     *
     * Callback signature:
     *   cb(data: bytes, width: int, height: int)
     *
     * `data` is the raw pixel buffer (RGB8, same as your ROS Image).
     */
    bool subscribe_image(const std::string &topic, py::function cb) {
        image_cb_ = std::move(cb);
        return node_.Subscribe<gz::msgs::Image>(
            topic,
            std::function<void(const gz::msgs::Image &)>(
                [this](const gz::msgs::Image &msg) { _on_image(msg); }
            ));
    }

    // ────────────────────────────────────────────────────────
    //  Publishers — advertise once, then publish by topic name
    // ────────────────────────────────────────────────────────

    bool advertise_twist(const std::string &topic) {
        pubs_[topic] = node_.Advertise<gz::msgs::Twist>(topic);
        return pubs_[topic].Valid();
    }

    void publish_twist(const std::string &topic,
                       double lx, double ly, double lz, double az) {
        gz::msgs::Twist msg;
        msg.mutable_linear()->set_x(lx);
        msg.mutable_linear()->set_y(ly);
        msg.mutable_linear()->set_z(lz);
        msg.mutable_angular()->set_z(az);
        pubs_.at(topic).Publish(msg);
    }

    bool advertise_bool(const std::string &topic) {
        pubs_[topic] = node_.Advertise<gz::msgs::Boolean>(topic);
        return pubs_[topic].Valid();
    }

    void publish_bool(const std::string &topic, bool value) {
        gz::msgs::Boolean msg;
        msg.set_data(value);
        pubs_.at(topic).Publish(msg);
    }

    bool advertise_empty(const std::string &topic) {
        pubs_[topic] = node_.Advertise<gz::msgs::Empty>(topic);
        return pubs_[topic].Valid();
    }

    void publish_empty(const std::string &topic) {
        gz::msgs::Empty msg;
        pubs_.at(topic).Publish(msg);
    }

    // ────────────────────────────────────────────────────────
    //  Lifecycle
    // ────────────────────────────────────────────────────────

    /**
     * Signal internal threads to stop invoking Python callbacks.
     * Call this before letting the object be garbage-collected.
     */
    void shutdown() {
        active_.store(false);
    }

private:
    // ── Callback trampolines (run on gz-transport threads) ──

    void _on_odom(const gz::msgs::Odometry &msg) {
        if (!active_.load()) return;

        const auto &p = msg.pose().position();
        const auto &q = msg.pose().orientation();
        const auto &v = msg.twist().linear();

        // Acquire the GIL so we can safely call into Python.
        // This is what lets the Python callback do things like
        // self._odom_event.set() from the gz-transport thread.
        py::gil_scoped_acquire gil;
        if (!odom_cb_.is_none()) {
            odom_cb_(p.x(), p.y(), p.z(),
                     q.w(), q.x(), q.y(), q.z(),
                     v.x(), v.y(), v.z());
        }
    }

    void _on_image(const gz::msgs::Image &msg) {
        if (!active_.load()) return;

        py::gil_scoped_acquire gil;
        if (!image_cb_.is_none()) {
            const auto &data = msg.data();
            image_cb_(py::bytes(data.data(), data.size()),
                      msg.width(), msg.height());
        }
    }

    // ── Members ──

    gz::transport::Node node_;
    std::unordered_map<std::string, gz::transport::Node::Publisher> pubs_;

    // Python callback objects.  Stored as py::object so we can
    // initialize them to py::none() without needing a callable.
    py::object odom_cb_ = py::none();
    py::object image_cb_ = py::none();

    // Flipped to false on shutdown() to prevent callbacks from
    // touching Python objects during teardown.
    std::atomic<bool> active_{true};
};


// ────────────────────────────────────────────────────────────
//  Module definition
// ────────────────────────────────────────────────────────────

PYBIND11_MODULE(gz_transport_py, m) {
    m.doc() = "Minimal Python bindings for gz-transport12 "
              "(Odometry, Image, Twist, Boolean, Empty)";

    py::class_<GzNode>(m, "GzNode")
        .def(py::init<>())
        // Subscriptions
        .def("subscribe_odom",  &GzNode::subscribe_odom,
             py::arg("topic"), py::arg("callback"),
             "Subscribe to an Odometry topic.  "
             "Callback: (px,py,pz, qw,qx,qy,qz, vx,vy,vz)")
        .def("subscribe_image", &GzNode::subscribe_image,
             py::arg("topic"), py::arg("callback"),
             "Subscribe to an Image topic.  "
             "Callback: (data_bytes, width, height)")
        // Publishers
        .def("advertise_twist",  &GzNode::advertise_twist,  py::arg("topic"))
        .def("publish_twist",    &GzNode::publish_twist,
             py::arg("topic"),
             py::arg("lx"), py::arg("ly"), py::arg("lz"), py::arg("az"))
        .def("advertise_bool",   &GzNode::advertise_bool,   py::arg("topic"))
        .def("publish_bool",     &GzNode::publish_bool,
             py::arg("topic"), py::arg("value"))
        .def("advertise_empty",  &GzNode::advertise_empty,  py::arg("topic"))
        .def("publish_empty",    &GzNode::publish_empty,    py::arg("topic"))
        // Lifecycle
        .def("shutdown",         &GzNode::shutdown);
}
