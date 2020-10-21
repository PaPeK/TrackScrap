# TrackScrap - Removes tracking errors and allows further processing

Tracking software as [idTracker](https://www.nature.com/articles/nmeth.2994) is not perfect and identity switches or other tracking errors can occur. Sometimes it is not even an error of the software but the camera did not save all frames.
This small package provides a framework to clean your data. It is minimal and mainly aims at identifying tracking errors and removing them to be able to work with reliable data. The used methods are probably not the best and any suggestion for improvement is welcome.

## Install

Install locally via `pip3 install -e . --user`

## Example

There exists an ipython-notebook example file `TrackScrapExample.ipynb` which will demonstrate some functionalities by cleaning idTracker trajectories of zebra-fish.
There is one function which exludes unexpected jumps.
However, it misses small jumps but the package can be extended with other functions very easily.

## User Agreement

By downloading TrackScrap you agree with the following points: TrackScrap is provided without any warranty or conditions of any kind. We assume no responsibility for errors or omissions in the results and interpretations following from application of TrackScrap.

## License

Copyright (C) 2017-2020 Pascal Klamser

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
