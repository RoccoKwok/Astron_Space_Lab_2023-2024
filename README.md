This is a program that calculate the velocity of a satellite by combining the **photo capturing and 3-axis accelerometer**.<br>
It used photo capturing to take some photos on the ISS(International Space Station). By calculating their distance difference in pixel, we can get the estimated velocity with a certain time.<br>
Beside, it also used 3-axis accelerometer to find the linear acceleration of the ISS(International Space Station) which calculated by **matrix**. And then **intergated the linear acceleration with calculus** and thus it will find the linear velocity.<br>
By combining them in one system, it can significantly decrease the error of miscalculating in **±0.1%**.
