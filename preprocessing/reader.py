import os
import cv2
import glob
from abc import ABC, abstractmethod


class Reader(ABC):
  @abstractmethod
  def read_microscope_data(self):
    pass

  @abstractmethod
  def read_biosensor_data(self):
    pass


class NanoReader(Reader):
  def __init__(self, base_path: str, flip_epic: list[bool]):
    self.base_path = base_path
    self.flip_epic = flip_epic

  def read_microscope_data(self):
    folder_path = os.path.join(self.base_path, "img_data")
    img_paths = glob.glob(os.path.join(folder_path, "*.jpeg"))

    result = {}
    for path in img_paths:
      name = os.path.splitext(os.path.basename(path))[0]
      data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
      result[name] = data
    
    return result
  
  def read_biosensor_data(self):
    from nanobio_core.epic_cardio.processing import load_data

    folder_path = os.path.join(self.base_path, "epic_data")
    raw_wells, _, _ = load_data(folder_path, flip=self.flip_epic)
    return raw_wells

