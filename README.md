CDICA 데이터셋 검증
최상위 디렉토리 구조 : ['CADICA']
시스템 내 감지된 총 이미지 프레임 수 : 31500 장
시스템 내 감지된 labeling 관련 텍스트 파일 수 : 4463 개
샘플 정답 파일 구조 확인 - readme.txt

<br>
 Please cite the following paper when using CADICA:
 Jiménez‐Partinen, A., Molina‐Cabello, M. A., Thurnhofer‐Hemsi, K., Palomo, E. J., Rodríguez‐Capitán, J., Molina‐Ramos, A. I., & Jiménez‐Navarro, M. (2024). CADICA: A new dataset for coronary artery disease detection by using invasive coronary angiography. Expert Systems, 41(12), e13708.
 
 The CADICA dataset is an annotated dataset of Invasive Coronary Angiography (ICA) images from 42 patients, including manually labeled lesion bounding boxes and selected clinical features. In ICA imaging, lesion degree assessment is commonly visually estimated, which implies a subjective factor and interobserver variability. Accurate identification of lesions is crucial for correct diagnoses and treatment. This motivates the development of computer-aided systems that can support specialists in their clinical procedures. This dataset can be utilized by clinicians to refine their skills in angiographic assessment of coronary artery disease (CAD) severity, by computer scientists to develop computer-aided diagnostic systems, and to validate existing CAD detection methods, thereby approaching solutions for clinical settings.
 

Please cite the following paper when using CADICA:
Jiménez‐Partinen, A., Molina‐Cabello, M. A., Thurnhofer‐Hemsi, K., Palomo, E. J., Rodríguez‐Capitán, J., Molina‐Ramos, A. I., & Jiménez‐Navarro, M. (2024). CADICA: A new dataset for coronary artery disease detection by using invasive coronary angiography. Expert Systems, 41(12), e13708.

The CADICA dataset is an annotated dataset of Invasive Coronary Angiography (ICA) images from 42 patients, including manually labeled lesion bounding boxes and selected clinical features. In ICA imaging, lesion degree assessment is commonly visually estimated, which implies a subjective factor and interobserver variability. Accurate identification of lesions is crucial for correct diagnoses and treatment. This motivates the development of computer-aided systems that can support specialists in their clinical procedures. This dataset can be utilized by clinicians to refine their skills in angiographic assessment of coronary artery disease (CAD) severity, by computer scientists to develop computer-aided diagnostic systems, and to validate existing CAD detection methods, thereby approaching solutions for clinical settings.

The CADICA dataset is a directory that contains the "metadata.xlsx" file, which stores the clinical data, as well as two main folders that differentiate the videos selected by the medical team for each patient: "nonselectedVideos" and "selectedVideos". Inside each folder, there are several sub-directories with the naming convention "pX", where X is the ID of each patient, and "vY", where Y is the ID of the video of that patient. 

The folder "pX" contains the following information: 
	"vY": several sub-directories with the videos selected for that patient. 
	"lesionVideos.txt": includes the IDs of the videos chosen where appears at least one lesion which is labeled.
	"nonlesionVideos.txt": contains the IDs of the selected videos with no visible lesions.

The folder "vY" contains the following information:
	"input": a sub-directory containing a separate PNG file for each video frame.
	"pX_vY_selectedFrames.txt": includes the IDs of the keyframes for the medical team for all the selected videos. 
	"groundtruth": a sub-directory available only if there are lesions in that selected video.

The folder "groundtruth" contains the following information:
	"pX_vY_000ZZ.txt": contains the bounding boxes and their category in each row. There are such files as frames in "pX_vY_selectedFrames.txt". Bounding boxes are specified in the format [x,y,w,h], where (x,y) are the pixel coordinates of the top left corner, w is the width, and h is the height of the bounding box.
	"pX_vY_groundTruthTable.mat": contains a table with the ground truth information of that video. 

Inside the "selectedVideos" folder, you can find the "CADICAprojections.json" file, containing video projections. Please note that the total number of videos may not match the total number of selected videos, as manual discarding can be done.
