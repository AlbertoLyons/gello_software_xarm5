from pathlib import Path
from typing import Optional, Union

from dm_control import mjcf

from gello.dm_control_tasks.arms.manipulator import Manipulator
from gello.dm_control_tasks.mjcf_utils import MENAGERIE_ROOT


class xArm5(Manipulator):
    GRIPPER_XML = None
    XML = MENAGERIE_ROOT / "ufactory_xarm5" / "xarm5_noarm.xml"

    def _build(
        self,
        name: str = "xArm5",
        xml_path: Union[str, Path] = XML,
        gripper_xml_path: Optional[Union[str, Path]] = GRIPPER_XML,
    ) -> None:
        super()._build(name=name, xml_path=xml_path, gripper_xml_path=gripper_xml_path)

    @property
    def flange(self) -> mjcf.Element:
        return self._mjcf_root.find("site", "attachment_site")
