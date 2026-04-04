"""
Generate ghost-drone Gazebo world for MPI parallel training.

Usage from CLI:
    python3 -m rl_uav_package.utils.generate_ghost_world --n 4

Usage from run.sh:
    python3 -c "from rl_uav_package.utils.generate_ghost_world import generate; generate(4)"

Generates:
    sim/models/tello_ghost_{0..N-1}/model.sdf
    sim/tello_world_{N}drone_ghost.sdf

Cleanup:
    python3 -c "from rl_uav_package.utils.generate_ghost_world import cleanup; cleanup(4)"
"""

import argparse
import os
import shutil
import textwrap

from rl_uav_package.config.constants import (
    SIM_CAMERA_WIDTH, SIM_CAMERA_HEIGHT, SIM_CAMERA_HFOV_RAD,
)


def _ghost_model_sdf(rank: int) -> str:
    """Generate a ghost drone model.sdf for the given MPI rank."""
    return textwrap.dedent(f"""\
        <?xml version="1.0" ?>
        <!-- Auto-generated ghost drone for rank {rank}. No visuals. -->
        <sdf version="1.8">
          <model name="tello_{rank}">
            <link name="base_link">
              <pose>0 0 0.0205 0 0 0</pose>
              <inertial>
                <mass>0.085</mass>
                <inertia>
                  <ixx>0.0000725</ixx>
                  <iyy>0.0000799</iyy>
                  <izz>0.0001286</izz>
                  <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
                </inertia>
              </inertial>

              <collision name="base_collision">
                <geometry><box><size>1e-10 1e-10 1e-10</size></box></geometry>
              </collision>

              <sensor name="front_camera" type="camera">
                <pose>0.049 0 0 0 0 0</pose>
                <camera>
                  <horizontal_fov>{SIM_CAMERA_HFOV_RAD}</horizontal_fov>
                  <image><width>{SIM_CAMERA_WIDTH}</width><height>{SIM_CAMERA_HEIGHT}</height><format>R8G8B8</format></image>
                  <clip><near>0.1</near><far>100</far></clip>
                </camera>
                <always_on>1</always_on>
                <update_rate>15</update_rate>
                <visualize>false</visualize>
                <topic>camera_{rank}/image_raw</topic>
              </sensor>

              <sensor name="downward_tof" type="gpu_lidar">
                <pose>0 0 -0.0205 0 1.570796 0</pose>
                <topic>tof_range_{rank}</topic>
                <update_rate>15</update_rate>
                <ray>
                  <scan>
                    <horizontal><samples>1</samples><resolution>1</resolution><min_angle>0</min_angle><max_angle>0</max_angle></horizontal>
                  </scan>
                  <range><min>0.05</min><max>3.0</max><resolution>0.01</resolution></range>
                </ray>
                <always_on>1</always_on>
                <visualize>false</visualize>
              </sensor>
            </link>

            <link name="rotor_1">
              <pose>0.063 -0.063 0.025 0 0 0</pose>
              <inertial>
                <mass>0.0005</mass>
                <inertia><ixx>1.82e-07</ixx><iyy>1.82e-07</iyy><izz>3.63e-07</izz></inertia>
              </inertial>
            </link>
            <joint name="rotor_1_joint" type="revolute">
              <child>rotor_1</child><parent>base_link</parent>
              <axis>
                <xyz>0 0 1</xyz>
                <limit><lower>-1e+16</lower><upper>1e+16</upper></limit>
                <dynamics><friction>0</friction></dynamics>
              </axis>
            </joint>

            <link name="rotor_2">
              <pose>0.063 0.063 0.025 0 0 0</pose>
              <inertial>
                <mass>0.0005</mass>
                <inertia><ixx>1.82e-07</ixx><iyy>1.82e-07</iyy><izz>3.63e-07</izz></inertia>
              </inertial>
            </link>
            <joint name="rotor_2_joint" type="revolute">
              <child>rotor_2</child><parent>base_link</parent>
              <axis>
                <xyz>0 0 1</xyz>
                <limit><lower>-1e+16</lower><upper>1e+16</upper></limit>
                <dynamics><friction>0</friction></dynamics>
              </axis>
            </joint>

            <link name="rotor_3">
              <pose>-0.063 0.063 0.025 0 0 0</pose>
              <inertial>
                <mass>0.0005</mass>
                <inertia><ixx>1.82e-07</ixx><iyy>1.82e-07</iyy><izz>3.63e-07</izz></inertia>
              </inertial>
            </link>
            <joint name="rotor_3_joint" type="revolute">
              <child>rotor_3</child><parent>base_link</parent>
              <axis>
                <xyz>0 0 1</xyz>
                <limit><lower>-1e+16</lower><upper>1e+16</upper></limit>
                <dynamics><friction>0</friction></dynamics>
              </axis>
            </joint>

            <link name="rotor_4">
              <pose>-0.063 -0.063 0.025 0 0 0</pose>
              <inertial>
                <mass>0.0005</mass>
                <inertia><ixx>1.82e-07</ixx><iyy>1.82e-07</iyy><izz>3.63e-07</izz></inertia>
              </inertial>
            </link>
            <joint name="rotor_4_joint" type="revolute">
              <child>rotor_4</child><parent>base_link</parent>
              <axis>
                <xyz>0 0 1</xyz>
                <limit><lower>-1e+16</lower><upper>1e+16</upper></limit>
                <dynamics><friction>0</friction></dynamics>
              </axis>
            </joint>

            <plugin filename="gz-sim-multicopter-motor-model-system"
                    name="gz::sim::systems::MulticopterMotorModel">
              <robotNamespace>model/tello_{rank}</robotNamespace>
              <jointName>rotor_1_joint</jointName>
              <linkName>rotor_1</linkName>
              <turningDirection>ccw</turningDirection>
              <maxRotVelocity>3000</maxRotVelocity>
              <motorConstant>4.22e-08</motorConstant>
              <momentConstant>0.016</momentConstant>
              <commandSubTopic>command/motor_speed</commandSubTopic>
              <rotorDragCoefficient>8.0e-06</rotorDragCoefficient>
              <rollingMomentCoefficient>1e-07</rollingMomentCoefficient>
              <rotorVelocitySlowdownSim>10</rotorVelocitySlowdownSim>
              <actuator_number>0</actuator_number>
              <motorType>velocity</motorType>
            </plugin>

            <plugin filename="gz-sim-multicopter-motor-model-system"
                    name="gz::sim::systems::MulticopterMotorModel">
              <robotNamespace>model/tello_{rank}</robotNamespace>
              <jointName>rotor_2_joint</jointName>
              <linkName>rotor_2</linkName>
              <turningDirection>cw</turningDirection>
              <maxRotVelocity>3000</maxRotVelocity>
              <motorConstant>4.22e-08</motorConstant>
              <momentConstant>0.016</momentConstant>
              <commandSubTopic>command/motor_speed</commandSubTopic>
              <rotorDragCoefficient>8.0e-06</rotorDragCoefficient>
              <rollingMomentCoefficient>1e-07</rollingMomentCoefficient>
              <rotorVelocitySlowdownSim>10</rotorVelocitySlowdownSim>
              <actuator_number>1</actuator_number>
              <motorType>velocity</motorType>
            </plugin>

            <plugin filename="gz-sim-multicopter-motor-model-system"
                    name="gz::sim::systems::MulticopterMotorModel">
              <robotNamespace>model/tello_{rank}</robotNamespace>
              <jointName>rotor_3_joint</jointName>
              <linkName>rotor_3</linkName>
              <turningDirection>ccw</turningDirection>
              <maxRotVelocity>3000</maxRotVelocity>
              <motorConstant>4.22e-08</motorConstant>
              <momentConstant>0.016</momentConstant>
              <commandSubTopic>command/motor_speed</commandSubTopic>
              <rotorDragCoefficient>8.0e-06</rotorDragCoefficient>
              <rollingMomentCoefficient>1e-07</rollingMomentCoefficient>
              <rotorVelocitySlowdownSim>10</rotorVelocitySlowdownSim>
              <actuator_number>2</actuator_number>
              <motorType>velocity</motorType>
            </plugin>

            <plugin filename="gz-sim-multicopter-motor-model-system"
                    name="gz::sim::systems::MulticopterMotorModel">
              <robotNamespace>model/tello_{rank}</robotNamespace>
              <jointName>rotor_4_joint</jointName>
              <linkName>rotor_4</linkName>
              <turningDirection>cw</turningDirection>
              <maxRotVelocity>3000</maxRotVelocity>
              <motorConstant>4.22e-08</motorConstant>
              <momentConstant>0.016</momentConstant>
              <commandSubTopic>command/motor_speed</commandSubTopic>
              <rotorDragCoefficient>8.0e-06</rotorDragCoefficient>
              <rollingMomentCoefficient>1e-07</rollingMomentCoefficient>
              <rotorVelocitySlowdownSim>10</rotorVelocitySlowdownSim>
              <actuator_number>3</actuator_number>
              <motorType>velocity</motorType>
            </plugin>

            <plugin filename="gz-sim-multicopter-control-system"
                    name="gz::sim::systems::MulticopterVelocityControl">
              <robotNamespace>model/tello_{rank}</robotNamespace>
              <commandSubTopic>cmd_vel</commandSubTopic>
              <enableSubTopic>enable</enableSubTopic>
              <comLinkName>base_link</comLinkName>
              <velocityGain>1.0 1.0 0.8</velocityGain>
              <attitudeGain>0.1 0.1 0.05</attitudeGain>
              <angularRateGain>0.03 0.03 0.02</angularRateGain>
              <rotorConfiguration>
                <rotor>
                  <jointName>rotor_1_joint</jointName>
                  <forceConstant>4.22e-08</forceConstant>
                  <momentConstant>0.016</momentConstant>
                  <direction>1</direction>
                </rotor>
                <rotor>
                  <jointName>rotor_2_joint</jointName>
                  <forceConstant>4.22e-08</forceConstant>
                  <momentConstant>0.016</momentConstant>
                  <direction>-1</direction>
                </rotor>
                <rotor>
                  <jointName>rotor_3_joint</jointName>
                  <forceConstant>4.22e-08</forceConstant>
                  <momentConstant>0.016</momentConstant>
                  <direction>1</direction>
                </rotor>
                <rotor>
                  <jointName>rotor_4_joint</jointName>
                  <forceConstant>4.22e-08</forceConstant>
                  <momentConstant>0.016</momentConstant>
                  <direction>-1</direction>
                </rotor>
              </rotorConfiguration>
            </plugin>

            <plugin filename="gz-sim-odometry-publisher-system"
                    name="gz::sim::systems::OdometryPublisher">
              <odom_frame>odom</odom_frame>
              <robot_base_frame>base_link</robot_base_frame>
              <dimensions>3</dimensions>
              <odom_publish_frequency>30</odom_publish_frequency>
            </plugin>
          </model>
        </sdf>
    """)


def _world_sdf(n_drones: int) -> str:
    """Generate the world SDF with one landing setup and N ghost drones."""
    drone_includes = ""
    for i in range(n_drones):
        y_offset = 0.1 * i  # slight spread so they don't stack at startup
        drone_includes += f"""
    <include>
      <uri>model://tello_ghost_{i}</uri>
      <pose>2.0 {y_offset:.1f} 1.0 0 0 3.14159</pose>
    </include>
"""

    return textwrap.dedent(f"""\
        <?xml version="1.0" ?>
        <!--
          Auto-generated ghost world: {n_drones} invisible drones, one landing setup.
          Generated by generate_ghost_world.py
        -->
        <sdf version="1.8">
          <world name="tello_sim">

            <physics name="fast_sim" type="ignored">
              <max_step_size>0.001</max_step_size>
              <real_time_factor>0</real_time_factor>
              <real_time_update_rate>0</real_time_update_rate>
            </physics>

            <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
            <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
            <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
            <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
              <render_engine>ogre2</render_engine>
            </plugin>

            <light type="directional" name="sun">
              <pose>0 0 10 0 0 0</pose>
              <diffuse>0.8 0.8 0.8 1</diffuse>
              <specular>0.2 0.2 0.2 1</specular>
              <direction>-0.5 0.1 -0.9</direction>
            </light>

            <model name="ground_plane">
              <static>true</static>
              <link name="link">
                <collision name="collision"><geometry><plane><normal>0 0 1</normal></plane></geometry></collision>
                <visual name="visual">
                  <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
                  <material><ambient>0.7 0.7 0.7 1</ambient><diffuse>0.7 0.7 0.7 1</diffuse></material>
                </visual>
              </link>
            </model>

            <model name="desk">
              <static>true</static>
              <pose>0 0 0.375 0 0 1.5708</pose>
              <link name="link">
                <collision name="collision"><geometry><box><size>0.60 1.00 0.75</size></box></geometry></collision>
              <visual name="visual">
                <geometry><mesh><uri>model://meshes/table.glb</uri></mesh></geometry>
              </visual>
              </link>
            </model>

            <model name="landing_pad">
              <static>true</static>
              <pose>0.05 0 0.75 0 0 0</pose>
              <link name="link">
              <visual name="visual">
                <geometry><cylinder><radius>0.15</radius><length>0.01</length></cylinder></geometry>
                <material><ambient>0.1 0.7 0.1 1</ambient><diffuse>0.1 0.7 0.1 1</diffuse></material>
              </visual>
              </link>
            </model>

            <model name="marker_pole">
              <static>true</static>
              <pose>-0.40 0 0.925 0 0 0</pose>
              <link name="link">
                <visual name="visual">
                  <geometry><cylinder><radius>0.005</radius><length>0.35</length></cylinder></geometry>
                  <material><ambient>0.05 0.05 0.05 1</ambient><diffuse>0.05 0.05 0.05 1</diffuse></material>
                </visual>
              </link>
            </model>

            <model name="aruco_marker">
              <static>true</static>
              <pose>-0.40 0 1.10 0 0 0</pose>
              <link name="link">
                <visual name="visual">
                  <geometry><box><size>0.01 0.20 0.20</size></box></geometry>
                  <material>
                    <ambient>1.0 1.0 1.0 1</ambient>
                    <diffuse>1.0 1.0 1.0 1</diffuse>
                    <emissive>1.0 1.0 1.0 1</emissive>
                    <pbr>
                      <metal>
                        <albedo_map>aruco_marker.png</albedo_map>
                        <emissive_map>aruco_marker.png</emissive_map>
                      </metal>
                    </pbr>
                  </material>
                </visual>
              </link>
            </model>

            <model name="marker_backplate">
              <static>true</static>
              <pose>-0.41 0 1.10 0 0 0</pose>
              <link name="link">
                <visual name="visual">
                  <geometry><box><size>0.005 0.22 0.22</size></box></geometry>
                  <material>
                    <ambient>0.02 0.02 0.02 1</ambient>
                    <diffuse>0.02 0.02 0.02 1</diffuse>
                  </material>
                </visual>
              </link>
            </model>

            <!-- ══ Ghost drones ══════════════════════════════════ -->
        {drone_includes}
          </world>
        </sdf>
    """)


def generate(n_drones: int, sim_dir: str = None) -> str:
    """Generate ghost world files. Returns path to the world SDF.

    Parameters
    ----------
    n_drones : int
        Number of ghost drones to generate.
    sim_dir : str, optional
        Path to the sim/ directory. If None, auto-detected relative
        to this file's location in the repo.

    Returns
    -------
    world_path : str
        Absolute path to the generated world SDF.
    """
    if sim_dir is None:
        # Navigate from rl_uav_package/utils/ up to repo root, then into sim/
        this_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(this_dir))
        sim_dir = os.path.join(repo_root, "sim")

    models_dir = os.path.join(sim_dir, "models")

    # Generate model directories
    for i in range(n_drones):
        model_dir = os.path.join(models_dir, f"tello_ghost_{i}")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model.sdf")
        with open(model_path, "w") as f:
            f.write(_ghost_model_sdf(i))
        config_path = os.path.join(model_dir, "model.config")
        with open(config_path, "w") as f:
            f.write(textwrap.dedent(f"""\
                <?xml version="1.0"?>
                <model>
                  <name>tello_ghost_{i}</name>
                  <version>1.0</version>
                  <sdf version="1.8">model.sdf</sdf>
                  <description>Auto-generated ghost drone {i}</description>
                </model>
            """))

    # Generate world SDF
    world_filename = f"tello_world_{n_drones}drone_ghost.sdf"
    world_path = os.path.join(sim_dir, world_filename)
    with open(world_path, "w") as f:
        f.write(_world_sdf(n_drones))

    print(f"[generate_ghost_world] Created {n_drones} ghost models + {world_filename}")
    return world_path


def cleanup(n_drones: int, sim_dir: str = None):
    """Remove generated ghost world files.

    Parameters
    ----------
    n_drones : int
        Number of ghost drone directories to remove.
    sim_dir : str, optional
        Path to the sim/ directory. Auto-detected if None.
    """
    if sim_dir is None:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(this_dir))
        sim_dir = os.path.join(repo_root, "sim")

    models_dir = os.path.join(sim_dir, "models")

    removed = 0
    for i in range(n_drones):
        model_dir = os.path.join(models_dir, f"tello_ghost_{i}")
        if os.path.isdir(model_dir):
            shutil.rmtree(model_dir)
            removed += 1

    world_path = os.path.join(sim_dir, f"tello_world_{n_drones}drone_ghost.sdf")
    if os.path.isfile(world_path):
        os.remove(world_path)

    print(f"[generate_ghost_world] Cleaned up {removed} ghost models + world SDF")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, required=True, help="Number of ghost drones")
    p.add_argument("--cleanup", action="store_true", help="Remove generated files")
    p.add_argument("--sim-dir", type=str, default=None)
    args = p.parse_args()

    if args.cleanup:
        cleanup(args.n, sim_dir=args.sim_dir)
    else:
        generate(args.n, sim_dir=args.sim_dir)