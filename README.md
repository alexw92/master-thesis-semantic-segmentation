# Tensorflow Framework

## Knowledge & Tutorials
+ Nice TensorFlow tutorial on [youtube](https://www.youtube.com/watch?v=yX8KuPZCAMo&lc=z12vwxxpxxqmjfwbv04cglyxixiuxrxp0aw) (about 50 min)

## Tensorflow installieren
Install Python 3.5.2 and include the installed folder into Path  (```C:\Wichtige Progs\Python``` in my case)

Install with ```pip``` according to the [docs](https://www.tensorflow.org/install/install_windows#installing_with_native_pip)

run the following testCode:

```python
import tensorflow as tf


def main():
    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    print(sess.run(hello))


# Call Main method
main()

```

# Python language knowledge

+ Why you need a ```self``` keyword in Python ? [Stackoverflow](https://stackoverflow.com/a/2725996)

+ There are no multiline comments in python apart from using triple ``` ''' multiline comment '''```
which isn't highlighted correctly

+ Python's [with statement](http://effbot.org/zone/python-with-statement.htm)

+ In general, in a statement such as ```Python
class Foo(Base1, Base2):```
Foo is being declared as a class inheriting from base classes Base1 and Base2.

+ [Python 3 Tutorial](https://docs.python.org/3/tutorial/)


# Pycharm tricks

+ Where to change key bindings:
```
File | Settings | Keymap
```

+ Where to activate documentation-popup on mouse-hover
```
File | Settings | Editor | Show quick doc on mouse move
```


