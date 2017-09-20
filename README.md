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

+ Print() Knowledge

If you don’t want characters prefaced by \ to be interpreted as special characters, you can use raw strings by adding an r before the first quote:
```Python
>>> print('C:\some\name')  # here \n means newline!
C:\some
ame
>>> print(r'C:\some\name')  # note the r before the quote
C:\some\name
```
Strings can be concatenated (glued together) with the + operator, and repeated with *:

```Python
>>> # 3 times 'un', followed by 'ium'
>>> 3 * 'un' + 'ium'
'unununium'
```

Break long strings (in Java you'd need "abcd"+"abc"+...):

```Python
>>> text = ('Put several strings within parentheses '
...         'to have them joined together.')
>>> text
'Put several strings within parentheses to have them joined together.'
```

Length of a string and cutting:
```Python
str = "abcde"
len( str ) #5
str[3]     #d
str[4]     #e

str[len(str)] #e
str[-1]       #e [Count from the end backwards]

lol = "Alexander"
lol[0:3]     #Ale
lol[:3]      #Ale
lol[3:]      #xander
```

Python string are *immutable*:
```Python
>>> word[0] = 'J'
  ...
TypeError: 'str' object does not support item assignment
```
___
### Lists

```Python
# Indexing like strings
numbers = [1,2,3,4,5]
numbers = numbers + [6,7,8,9]
>>> numbers[0]    # 1 indexing returns the item
>>> numbers[-3:]  # [3,4,5] slicing returns a new list

# Unlike strings, lists are a mutable type
cubes = [1, 8, 27, 65, 125]
cubes[3] = 64
>>> cubes              # [1, 8, 27, 64, 125]
cubes.append(216)  # add the cube of 6
>>> cubes              ## [1, 8, 27, 64, 125, 216]

# Slizing is also possible
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
>>> letters # ['a', 'b', 'c', 'd', 'e', 'f', 'g']
# replace some values
>>> letters[2:5] = ['C', 'D', 'E']
>>> letters # ['a', 'b', 'C', 'D', 'E', 'f', 'g']
# now remove them
>>> letters[2:5] = []
>>> letters # ['a', 'b', 'f', 'g']
# clear the list by replacing all the elements with an empty list
letters[:] = []
>>> letters # []
# len() works with lists as well
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
>>> len(letters)  # 3

# Nested lists
a = ['a', 'b', 'c']
n = [1, 2, 3]
x = [a, n]
>>> x  # [['a', 'b', 'c'], [1, 2, 3]]
>>> x[0] # ['a', 'b', 'c']
>>> x[0][1] # 'b'

```

___
### Control Structures

Fibonacci example:
```Python
>>> a, b = 0, 1
>>> while b < 1000:
...     print(b, end=',')
...     a, b = b, a+b
```

if-statement:
```Python
if x < 0:
    x = 0
    print('Negative changed to zero')
elif x == 0:
    print('Zero')
elif x == 1:
    print('Single')
else:
    print('More')
```

for-loop
```Python
# Iterating over a copy in order to keep loop range
for w in words[:]:  # Loop over a slice copy of the entire list.
    if len(w) > 6:
        words.insert(0, w)
        
# loop over range
for i in range(5):
    print(i)      # prints numbers from 0 to 4

range(5, 10)      # range from 5 to 9

range(0, 10, 3)   # 0 to 9 in 3-steps: 0, 3, 6, 9

# Iterate with indices
a = ['Mary', 'had', 'a', 'little', 'lamb']
for i in range(len(a)):
    print(i, a[i])
    
# Create List from range (Matlab style :D)
list(range(5)) # [0, 1, 2, 3, 4, 4]

# else in for-loop
# else is entered if loop range is exhausted without break 
for n in range(2, 10):
    for x in range(2, n):
        if n % x == 0:
            print(n, 'equals', x, '*', n//x)
            break
     else:
        # loop fell through without finding a factor
        print(n, 'is a prime number')
```

The *pass* statement does nothing. It can be used when a statement is required syntactically but the program requires no action. **Remember:** There are no semicolons ";" in python. 
```Python
while True:
    pass  # Busy-wait for keyboard interrupt (Ctrl+C)

class MyEmptyClass:
    pass
    
def initlog(*args):
    pass   # Remember to implement this!
```

___

Functions:
```Python
4.6
```

# Pycharm tricks

+ Where to change key bindings:
```
File | Settings | Keymap
```

+ Where to activate documentation-popup on mouse-hover
```
File | Settings | Editor | Show quick doc on mouse move
```


