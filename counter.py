import os
import torch

class IDCounter(object):
  def __init__(self, threshold, ID, sessionId, output_dir):
    self.count_thresh = threshold

    self.dir = output_dir
    self.sessionId = sessionId
    self.path = os.path.join(output_dir, f"{sessionId}.txt")
    self.ckpt = os.path.join(output_dir, f"backup-{sessionId}.pt")
    
    self.names = [f"{entity['first']} {entity['last']}" for entity in ID]

    with open(self.path, 'wb') as f:
      f.write(f"Attendance report for {self.sessionId}\n\n".encode("utf-8"))

    if os.path.exists(self.ckpt): 
      bundle = torch.load(self.ckpt)
      self.counters, self.attendance = bundle['counters'], bundle['attendance']
    
    else:
      self.counters = {i:0 for i in self.names}
      self.attendance = {i:False for i in self.names}
  

  def write(self, val):
    with open(self.path, 'ab') as f:
      f.write(val.encode("utf-8"))
  

  def registered_string(self, id):
    return f"✔\t{id}\n"


  def unregistered_string(self, id):
    return f"❌\t{id}\n"
  

  def update(self, id):
    if id in self.counters:
      self.counters[id] += 1
      if self.counters[id] >= self.count_thresh:
        if self.attendance[id] == False:
          self.attendance[id] = True
          torch.save({'counters': self.counters, 'attendance': self.attendance}, self.ckpt)
        return True
    else:
      self.counters[id] = 1
    return False

  def showReport(self):    
    append = "Registered:\n"
    for person in self.attendance:
      if self.attendance[person] == True:
        append += self.registered_string(person)
    self.write(append)

    append = f"\nNot registered:\n"
    for person in self.attendance:
      if self.attendance[person] == False:
        append += self.unregistered_string(person)
    append += '\n\n[TECLARS ATTENDANCE REPORT COMPLETE]\n'

    self.write(append)
    
    os.remove(self.ckpt)
    os.system(f"notepad.exe {self.path}")
