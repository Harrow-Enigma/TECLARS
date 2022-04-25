import os

from sympy import Id

class IDCounter(object):
  def __init__(self, threshold, ID, sessionId, path):
    self.count_thresh = threshold
    self.names = [f"{entity['first']} {entity['last']}" for entity in ID]
    self.counters = {i:0 for i in self.names}
    self.attendance = {i:False for i in self.names}

    self.sessionId = sessionId
    self.path = path
    with open(path, 'wb') as f:
      f.write(f"Attendance report for {sessionId}\n\nRegistered:\n".encode("utf-8"))
  
  def write(self, val):
    with open(self.path, 'ab') as f:
      f.write(val.encode("utf-8"))
  
  def update(self, id):
    if id in self.counters:
      self.counters[id] += 1
      if self.counters[id] >= self.count_thresh:
        if self.attendance[id] == False:
          self.attendance[id] = True
          self.write(f"✔\t{id}\n")
        return True
    else:
      self.counters[id] = 1
    return False

  def showReport(self):
    append = f"\nNot registered:\n"
    for person in self.attendance:
      if self.attendance[person] == False:
        append += f"❌\t{person}\n"
    append += '\n\n[TECLARS ATTENDANCE REPORT COMPLETE]\n'

    self.write(append)
    
    os.system(f"notepad.exe {self.path}")
