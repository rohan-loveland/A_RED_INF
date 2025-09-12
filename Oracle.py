"""# Oracle"""

class Oracle:
  def __init__(self, X, y):
    self.X = X #[[data]]
    self.y = y #[[label, relevant]]

  def answer_query(self, abs_index):
    if (abs_index >= self.X.shape[0]):
      print("Index out of range", abs_index, self.X.shape[0])
    label = self.y[abs_index][0]
    relevance = self.y[abs_index][1]
    return label, relevance