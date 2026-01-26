
# ADSA AI 2025 Fall Internship ML Project - [Portfolio Link](https://believed-station-bd6.notion.site/ML-Research-Project-Droplet-Analysis-2f4e909a031b8067aca1e0b31f5c2b7d?source=copy_link)


The first folder, DataScripts contain all the scripts used to create and manage the dataset used to train the CNN model.  
The second folder ModelScripts contains all the scripts used to create and train the CNN model using keras, tensorflow and slurm scripts(super computer job manager).


What? 
* Developed a CNN model that was able to determine 4 physical properties of a droplet from its outline.
- Generated 300,000+ labelled droplet images from real experimental images, using automated Python scripts.
- Learned and utilized the supercomputing clusters at the University of Hawaii.

Why? 
* Current Axisymmetric Drop Shape Analysis (ADSA) takes about 50ms to determine these properties. 
* Machine learning was implemented as a faster way to determine these properties.

Results:
* Created 4 models that had ~98% accuracy and 10 ms prediction time. 


<img width="600" height="600" alt="pred_vs_true_Vol" src="https://github.com/user-attachments/assets/62a3934f-a49d-4a15-af4b-fae571e2275a" />
<img width="600" height="600" alt="pred_vs_true_Area" src="https://github.com/user-attachments/assets/ac60a52b-2b13-43cb-872a-be9dad2b38cc" />
<img width="600" height="600" alt="pred_vs_true_Tension" src="https://github.com/user-attachments/assets/241aaef7-ace8-4855-a265-b02321dd3cb4" />
<img width="600" height="600" alt="pred_vs_true_curv" src="https://github.com/user-attachments/assets/fcf24a76-84c4-4225-b52f-7c2aaec2adc0" />

