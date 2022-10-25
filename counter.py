import os
import torch
from PIL import Image

class IDCounter(object):
  def __init__(self, threshold, ID, sessionId, output_dir):
    self.count_thresh = threshold

    self.dir = os.path.join(output_dir, sessionId)
    self.unk_dir = os.path.join(self.dir, 'unknowns')
    os.makedirs(self.dir, exist_ok=True)
    self.sessionId = sessionId
    self.path = os.path.join(self.dir, f"ATTENDANCE.txt")
    self.ckpt = os.path.join(self.dir, f"backup.pt")
    
    self.names = [f"{entity['first']} {entity['last']}" for entity in ID]

    if os.path.exists(self.ckpt): 
      bundle = torch.load(self.ckpt)
      self.counters, self.attendance, self.unk_count = bundle['counters'], bundle['attendance'], bundle['unk']
    
    else:
      self.counters = {i:0 for i in self.names}
      self.attendance = {i:False for i in self.names}
      self.unk_count = 0
  
  
  def registered_string(self, id):
    return f"✔\t{id}\n"

  def unregistered_string(self, id):
    return f"❌\t{id}\n"
  
  def make_report(self):    
    report = f"Attendance report for {self.sessionId}\n\n"
    report += "Registered:\n"
    for person in self.attendance:
      if self.attendance[person] == True:
        report += self.registered_string(person)

    report += f"\nNot registered:\n"
    for person in self.attendance:
      if self.attendance[person] == False:
        report += self.unregistered_string(person)
    report += '\n\n[TECLARS ATTENDANCE REPORT COMPLETE]\n'

    print(report, self.path)

    with open(self.path, 'wb') as f:
      f.write(report.encode("utf-8"))
  
  def save_ckpt(self):
    torch.save({'counters': self.counters, 'attendance': self.attendance, 'unk': self.unk_count}, self.ckpt)
  

  def save_unk_img(self, imgarr):
    os.makedirs(self.unk_dir, exist_ok=True)
    self.unk_count += 1
    self.save_ckpt()
    im = Image.fromarray(imgarr)
    im.save(os.path.join(self.unk_dir, f"unknown_{self.unk_count}.jpg"))
  

  def update(self, id):
    if id in self.counters:
      self.counters[id] += 1
      if self.counters[id] >= self.count_thresh:
        if self.attendance[id] == False:
          self.attendance[id] = True
          self.save_ckpt()
          self.make_report()
        return True
    else:
      self.counters[id] = 1
    return False  
