#three arguments are required
if [ "$#" -lt 3 ]; then
	echo "Usage: $0 input_video output_directory [frame_rate] [width] [height]"
	echo "If frame_rate is not provided, the video's default frame rate will be used."
	echo "If width and height are not provided, the video's default resolution will be used."
	exit 1
fi

# Assign input arguments to variables
input_video=$1
output_directory=$2
frame_rate=${3:-0}   # Default frame rate is 0 (use video's default)
width=${4:-null}     # Default width is null (use video's default)
height=${5:-null}    # Default height is null (use video's default)

# Create output directory if it doesn't exist
mkdir -p "$output_directory"

# Build the ffmpeg scale filter string
scale_filter=""
if [ "$width" != "null" ] && [ "$height" != "null" ]; then
	scale_filter=",scale=$width:$height"
fi

if [ "$frame_rate" -eq 0 ]; then
	ffmpeg -i "$input_video" -vf "fps=ntsc$scale_filter" -start_number 0 "$output_directory/%05d.png"
else
	ffmpeg -i "$input_video" -vf "fps=$frame_rate$scale_filter" -start_number 0 "$output_directory/%05d.png"
fi

echo "Frames extracted to $output_directory"
