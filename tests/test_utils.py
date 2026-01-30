from imas_muscle3.utils import increment_suffix

def test_incerement_suffix():
  assert increment_suffix('my_string') == 'my_string_1'
  assert increment_suffix('my_string_1') == 'my_string_2'
  assert increment_suffix('my_string_99') == 'my_string_100'
  assert increment_suffix('my_string_09') == 'my_string_10'
  assert increment_suffix('my_1_string') == 'my_1_string_1'
  assert increment_suffix('my_1_string_2') == 'my_1_string_3'
  assert increment_suffix('9') == '9_1'
  assert increment_suffix('_9') == '_10'
