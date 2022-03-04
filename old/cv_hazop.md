Risk Id|Location|Guide Word|Parameter|Meaning|Consequence|Risk
-------|--------|----------|--------|-------|-----------|------
0|Light Sources|No (not none)|Number|No light sources|No light available|Sensor will receive no light, but thermal noise or black current can cause wrong input
123|Light Sources|No (not none)|Intensity |Light Source is off|no light from this|Light Source in scene|Underexposure if no|other light sources on
124|Light Sources|No (not none)|Intensity| Light Source is off| Captured camera noise|leads to fake effects
125|Light Sources|More (more of, higher)|Intensity|Light Source is too strong|Too much light in scene|Overexposure of lit objects
126|Light Sources|As well as|Intensity |Strong and weak light sources mixed|Weak light sources are outshined by strong ones|If light source should be detected by CV algorithm, this may be hampered
127|Light Sources|As well as|Intensity|Strong and weak light sources mixed| Weak light sources are outshined by strong ones *| Might exceed the intensity range of sensor
128|Light Sources|Other than|Intensity|Intensity of light source is completely different than expected by application|Scene too bright or too dim|Over or underexposure of relevant objects or scene elements
