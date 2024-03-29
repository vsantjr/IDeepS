# Project IDeepS

<img src="https://github.com/vsantjr/CAP/blob/master/Images/logo2ideeps.png" width=58%>

## Login and Transference

Click [here](../Utils/ut.sh) to download the shell script to login and transfer files from/to SDumont. This script was tested on a Mac computer. It is interesting to add this file in the $PATH variable of your computer.


After connecting to the VPN, run in the terminal to login into your account:

```
ut.sh -c 0
```

In order to transfer a file to SDumont, run in the terminal:

```
ut.sh -t filename
```
Thus to transfer the file ```test.py```:

```ut.sh -t test.py```

In order to receive a file from SDumont, run in the terminal:

```
ut.sh -f filename
```

Thus to receive the file ```test.py```:

```ut.sh -f test.py```

In order to receive an entire dir from SDumont, run in the terminal:

```
ut.sh -d subdir
```

Thus to receive the dir ```img``` which is a subdirectory of the ```work``` directory in your SDumont account:

```
ut.sh -d img
```




## Author

[**Valdivino Alexandre de Santiago J&uacute;nior**](https://www.linkedin.com/in/valdivino-alexandre-de-santiago-j%C3%BAnior-103109206/?locale=en_US)

## Licence

This project is licensed under the GNU GENERAL PUBLIC LICENSE, Version 3 (GPLv3) - see the [LICENSE.md](../LICENSE) file for details.

## Cite

Please cite this repository if you use it as:

V. A. Santiago J&uacute;nior. IDeepS, 2024. Acessed on: *date of access*. Available at: https://github.com/vsantjr/IDeepS. 