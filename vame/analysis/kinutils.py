from collections import OrderedDict
import itertools

# import Preprocess.preprocess as pp

import ffmpeg
import glob
import logging
import numpy as np
import os
import pandas as pd
import re
import subprocess
import sys
import tempfile
import traceback
import yaml

from ipywidgets import Video, interact


class KinVideo:
    def __init__(
        self,
        filename=None,
        view=None,
        tfile=None,
        convertprocess=None,
        offset=0,
        fps=120,
    ):
        """
        Create a video instance for manipulation.

        filename - The file name (if tfile is not used).
        tfile - NamedTemporaryFile like object (filename is ignored).
        """
        if tfile is None and not os.path.isfile(filename):
            raise FileNotFoundError(f"File {filename} not found.")

        self.tfile = tfile
        if not (self.tfile is None):
            filename = self.tfile.name

        self.filename = filename
        self.view = view
        self.convertprocess = convertprocess
        self.offset = self.str2sec(offset) if type(offset) is str else offset
        return

    def probevid(self):
        """
        Probe the video parameters and set them in the instance.
        """
        probe = ffmpeg.probe(self.filename)
        # Get first video stream.
        vidinfo = next(
            stream for stream in probe["streams"] if stream["codec_type"] == "video"
        )
        self.width = int(vidinfo["width"])
        self.height = int(vidinfo["height"])
        self.nframes = int(vidinfo["nb_frames"])
        self.framerate = vidinfo["r_frame_rate"]
        f, b = [int(n) for n in self.framerate.split("/")]
        self.fps = np.round(f / b)
        return

    def filter_scale(self, convertprocess, args):
        """
        Add scale filter to the FFMPEG process.

        Parameters:
        convertprocess - The FFMPEG process (after input before output).
        args - Scale arguments (a number or an array-like of two fx, fy).

        Returns:
        The modified convertprocess.
        """
        if not hasattr(args, "__len__"):
            args = [args]

        if len(args) == 1:
            zoom = (args[0], args[0])
        elif len(args) == 2:
            zoom = args
        else:
            raise Exception(f"Invalid zoom specification more than two factors {args}")

        width = (self.width * zoom[0] // 2) * 2
        height = (self.height * zoom[1] // 2) * 2
        return convertprocess.filter_("scale", f"{width}", f"{height}")

    def filter_fps(self, convertprocess, args):
        """
        Change FPS by dropping or duplicating frames.

        Parameters:
        convertprocess - The FFMPEG process (after input before output).
        args - FPS arguments one of:
               1.
        """
        if type(args) != str:
            args = f"factor:{args}"

        separgs = args.split(":")
        if len(separgs) == 1:
            separgs = f"factor:{args}"

        if len(separgs) == 2:
            if separgs[0] == "factor":
                fps = self.fps * float(separgs[1])
            elif separgs[0] == "set":
                fps = float(separgs[1])
            else:
                raise Exception(f'Unkonwn argument "{separgs [0]}"')
        else:
            fps = float(args)

        return convertprocess.filter_("fps", f"{fps}")

    def filter_speed(self, convertprocess, args):
        """
        Change playback speed.

        Parameters:
        convertprocess - The FFMPEG process (after input before output).
        args - Speed up factor (or slowdown for <1)
        """
        return convertprocess.filter_("setpts", f"{1/args}*PTS")

    def clip(self, start=0, length=1, filename=None, **kwargs):
        """
        Returns a clip of the video.

        Arguments:
        start - Start position in seconds, or a string of the format
                hh:mm:ss.msec.
        len - Length of clip in seconds or a string as with start.

        Returns:
        A KinVideo instance.
        """
        self.waitfinished()
        if type(start) is str:
            start = self.str2sec(start)

        if type(length) is str:
            length = self.str2sec(length)

        start = start + self.offset / self.fps
        startstr = self.sec2str(start)
        lengthstr = self.sec2str(length)

        tfile = None
        args = ""
        if filename is None:
            tfile = tempfile.NamedTemporaryFile(suffix=".mp4")
            filename = tfile.name

        print(f"Converting video from {startstr} for {lengthstr} to {filename}")
        convertprocess = ffmpeg.input(self.filename, ss=startstr, t=lengthstr)
        for f, args in kwargs.items():
            filtername = f"filter_{f}"
            if hasattr(self, filtername):
                func = getattr(self, filtername)
                convertprocess = func(convertprocess, args)
            else:
                logging.warning(f"Unkonwn filter {f}")

        convertprocess = convertprocess.output(filename)
        if not tfile is None:
            convertprocess = convertprocess.overwrite_output().run_async()
        #        cmd = f'ffmpeg{args} -i {self.filename} -ss {start} -t {length} {filename}'
        if tfile is None:
            return KinVideo(
                filename=filename,
                convertprocess=convertprocess,
                offset=self.offset + start,
            )
        else:
            return KinVideo(
                tfile=tfile, convertprocess=convertprocess, offset=self.offset + start
            )

    def clip2(self, start=0, length=1, filename=None):
        """
        Returns a clip of the video.

        Arguments:
        start - Start position in seconds, or a string of the format
                hh:mm:ss.msec.
        len - Length of clip in seconds or a string as with start.

        Returns:
        A KinVideo instance.
        """
        if type(start) is str:
            start = self.str2sec(start)

        if type(length) is str:
            length = self.str2sec(length)

        start = self.sec2str(start)
        length = self.sec2str(length)

        tfile = None
        args = ""
        if filename is None:
            tfile = tempfile.NamedTemporaryFile(suffix=".mp4")
            filename = tfile.name
            args += " -y"

        cmd = f"ffmpeg{args} -i {self.filename} -ss {start} -t {length} {filename}"
        convertprocess = subprocess.Popen(
            cmd.split(" "), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        if tfile is None:
            return KinVideo(
                filename=filename,
                convertprocess=convertprocess,
                offset=self.offset + start,
                fps=self.fps,
            )
        else:
            return KinVideo(
                tfile=tfile,
                convertprocess=convertprocess,
                offset=self.offset + start,
                fps=self.fps,
            )

    def getvideo(self, wait=True):
        if not self.waitfinished(wait):
            return None

        return Video.from_file(self.filename)

    def isfinished(self):
        if self.convertprocess is None:
            return True

        return not (self.convertprocess.poll() is None)

    def waitfinished(self, wait=True):
        if not (self.convertprocess is None):
            waittime = None if wait else 0
            try:
                rc = self.convertprocess.wait(waittime)
            except TimeoutExpired:
                return False

            if not (rc is None):
                print(f"RC is {rc}")
                code = os.WEXITSTATUS(rc)
                if code != 0:
                    raise Exception(f"ffmpeg finished with error {code}")

            print("Job is done")
            self.convertprocess = None

        if not hasattr(self, "fps"):
            self.probevid()
        return True

    def sec2comps(self, t):
        hours = int(t / 3600)
        t = t - hours * 3600
        minutes = int(t / 60)
        t = t - minutes * 60
        seconds = int(t)
        t = t - seconds
        ms = int(t * 1000)
        t = t - ms
        return [hours, minutes, seconds, ms, t]

    def str2sec(self, t):
        scomps = t.split(":")
        if len(scomps) == 1 and scomps[0] == "" or len(scomps) > 3:
            raise ValueError(f'Invalid time string "{t}".')

        comps = [0] * 2 + [int(comp) for comp in scomps[:-1]] + [float(scomps[-1])]
        comps = comps[-3:]
        secs = comps[0] * 3600 + comps[1] * 60 + comps[2]
        return secs

    def sec2str(self, t):
        hours, minutes, seconds, ms, _ = self.sec2comps(t)
        return f"{hours}:{minutes:02d}:{seconds:02d}.{ms:03d}"

    def getoffset(self):
        return self.offset

    def getfps(self):
        return self.fps

    def getnframes(self):
        return self.nframes

    def __exit__(self):
        if not (self.tfile is None):
            self.tfile.close()

        return

    @classmethod
    def classinit(cls):
        if not hasattr(cls, "fs"):
            cls.fs = KinFS()

        if not hasattr(cls, "synctab"):
            cls.synctab = pd.read_csv(cls.fs.getsyncfile())

        return

    @classmethod
    def videofromsubject(cls, subject, date, view, offset=None, **kwargs):
        cls.classinit()
        matched = cls.synctab[
            (cls.synctab["rat"] == subject) & (cls.synctab["date"] == date)
        ]

        if offset == None:
            offset = 0
            if view == "Down":
                column = "rec_start_frame"
            elif view == "Up":
                column = "up_LED_frame"
            else:
                logging.warning(f"Unknown view type {view} - " + "offset default Down.")
                column = "rec_start_frame"

            if len(matched) >= 1:
                if len(matched) > 1:
                    logging.warning(f"Multiple entries for {subject}:{date}")
                offset = matched.iloc[0]["rec_start_frame"]
            else:
                logging.warning(f"No offset entry found for {subject}:{date}")

        filenames = cls.fs.getvideos(subject, date, view)
        if len(filenames) > 1:
            logging.warning(f"Multiple videos for {subject}:{date}:{view}. Using 1st.")
        elif len(filenames) == 0:
            raise Exception(f"No video found for {subject}:{date}:{view}.")

        return KinVideo(filenames.popitem()[1], view, offset=offset, **kwargs)


class KinFrameChooser:
    def __init__(self, clip, scale=1.0):
        self.clip = clip
        self.scale = scale
        self.process = None
        self.frames = None

        self.lasttime = None
        self.clip.waitfinished()
        self.probeclip()
        self.extractframes()
        return

    def extractframes(self, wait=True):
        self.process = (
            ffmpeg.input(clip.filename)
            .output("pipe:", format="rawvideo", pix_fmt="rgb24")
            .run_ascync(capture_stdout=True)
        )
        return

    def waitfinished(self, timeout=None):
        if self.proces is None:
            return True

        try:
            out, err = self.process.communicate(timeout=timeout)
        except TimeoutExpired as e:
            return False

        self.process = None
        self.frames = np.frombuffer(out, np.uint8).reshape(
            [self.nframes, height, width, 3]
        )
        return True

    def getchooser(self):
        self.waitfinished()
        if self.lasttime is None:
            self.lasttime = self.clip.sec2comps(self.offset)

        return


class KinFS:
    """
    Encapsulate file access to Themis files.
    """

    CONFIGFILE = "~/.themis.yaml"
    DEFAULTTOPDIR = "/Volumes/Shared/Data"

    def __init__(self, config=None):
        """
        Initialise the class.

        Parameters:
        config - Path of config file (default: None accesses ~/.themis
        """
        self.readconfig(config)
        return

    """
    Read the configuration file.

    Parameters:
    config - A dictionary with overriding values.
             RawData - Directory of raw data under the top directory.
             Analysis - Directory of analysis data under the top directory.
    """

    def readconfig(self, config=None):
        self.localtopdir = None
        try:
            filepath = os.path.expanduser(KinFS.CONFIGFILE)
            with open(filepath) as cfile:
                self.config = yaml.load(cfile, Loader=yaml.loader.SafeLoader)

                self.topdir = self.tofilelist(self.config["TopDir"])
                if "LocalTopDir" in self.config:
                    self.localtopdir = self.tofilelist(self.config["LocalTopDir"])
        except Exception as e:
            traceback.print_exception(*sys.exc_info())

            logging.error(f'No configuration file found in "{filepath}".')
            self.topdir = [os.path.expanduser(KinFS.DEFAULTTOPDIR)]
            self.config = None

        if self.config is None:
            self.rawdata = "Video"
            self.analysis = "Analysis"
        else:
            self.rawdata = self.config["RawData"]
            self.analysis = self.config["Analysis"]

        return

    def tofilelist(self, value):
        if type(value) is str:
            value = [value]

        return [os.path.expanduser(cfile) for cfile in value]

    def searchinpath(self, path, subpath):
        """
        Return the first occurance of a supath in a list of paths.

        Parameters:
        path - List of directories to search.
        subpath - path of the file in one of the listed directories.

        Returns:
        Path to the file.
        """
        for directory in path:
            filepath = os.path.join(directory, subpath)
            if os.path.exists(filepath):
                return filepath

        return None

    def filesinpath(self, path, subpath):
        return (
            os.path.join(dir, subpath)
            for dir in path
            if os.path.exists(os.path.join(dir, subpath))
        )

    def getallsearchpaths(self, searchpath, subpath):
        """
        Return all the search paths of a supath in a list of paths.

        Parameters:
        path - List of directories to search.
        subpath - path of the file in one of the listed directories.

        Returns:
        Path to the file.
        """
        return [
            filepath
            for filepath in [
                os.path.join(directory, subpath) for directory in searchpath
            ]
            if os.path.exists(filepath)
        ]

    def gettopdirs(self):
        return [cdir for cdir in self.localtopdir + self.topdir if not (cdir is None)]

    def getrawpaths(self, subject, date, local=True, make=False):
        """
        Return the paths of experiment raw data.

        Parameters:
        subject - Name of subject.
        date - Experiment date.
        local - Whether to search just in the local path or in the remote path.
        make - Create directory if none is found (in the local directory).

        Returns:
        Path to experiment data.
        """
        dirs = self.localtopdir
        if not local:
            dirs = dirs + self.topdir
        subpath = os.path.join(self.rawdata, subject, date)
        paths = self.getallsearchpaths(dirs, subpath)
        if len(paths) == 0 and make:
            paths = [os.path.join(self.localtopdir[0], subpath)]
            os.makedirs(paths[0])
        return paths

    def getvideos(self, subject, date, view):
        """
        Searches for videos of format gx####.mp4 or ####.mp4 in the search path
        (local than remote if exist), and returns an OrderedDict with the most
        favourable videos first. The key is the video name (####) and the value
        is the path.

        subject - Name of subject.
        date - Experiment date.

        Returns:
        Dictionary of videos. The key is the video name (####) and the value is
        its path. First item is the most favourable.
        """
        viewpaths = self.getallsearchpaths(
            self.gettopdirs(), os.path.join(self.rawdata, subject, date, view)
        )
        videos = [
            glob.glob(os.path.join(directory, f"{pfx}[0-9]*.{sfx}"))
            for directory in viewpaths
            for pfx in ["", "gx"]
            for sfx in ["mp4", "MP4"]
        ]
        videos = itertools.chain(*videos)
        #
        # Filter out double entries.
        #
        videodict = OrderedDict()
        for path in videos:
            videoname = re.match("[^0-9]*([0-9]+)\..*", os.path.basename(path))
            if videoname is None:
                continue

            if not videoname.groups()[0] in videodict:
                videodict[videoname.groups()[0]] = path

        return videodict

    def getkinematicfiles(self, subject, date, view="WL"):
        """
        Searches for the kinematics files and returns the files sorted by file
        number.

        Parameters:
        subject - Name of subject.
        date - Experiment date.
        view - The view in which to search ('WL' is the default).

        Returns:
        An OrderedDict with the file number (according to its name) and the file
        path.
        """
        viewpaths = self.getallsearchpaths(
            self.gettopdirs(), os.path.join(self.rawdata, subject, date, view)
        )

        data = [
            glob.glob(os.path.join(directory, f"{pfx}[0-9]*.DT2"))
            for directory in viewpaths
            for pfx in ["NEUR", "BACK"]
        ]
        data = itertools.chain(*data)
        #
        # Sort on file number.
        #
        datafiles = {}
        first = 0
        last = -1
        for path in data:
            name = re.match("[^0-9]*([0-9]+)\.DT2", os.path.basename(path))
            if name is None:
                continue

            name = name.groups()[0]
            if name in datafiles:
                continue

            if int(name) > last:
                if last == -1:
                    first = int(name)

                last = int(name)

            if int(name) < first:
                first = int(name)

            datafiles[name] = path

        if last != -1 and len(datafiles) < last - first:
            logging.error(f"Missing data files.")

        return OrderedDict([(key, datafiles[key]) for key in sorted(datafiles.keys())])

    def getalldir(
        self,
        dirname=None,
        local=True,
        subject=None,
        date=None,
        view=None,
        create=False,
        test=True,
        anydir=False,
    ):
        topdir = None
        if local:
            topdir = self.localtopdir

        if topdir is None:
            topdir = self.topdir

        if anydir:
            topdir = self.gettopdirs()

        if not dirname is None:
            adir = (os.path.join(cdir, dirname) for cdir in topdir)
        else:
            adir = topdir

        if create:
            adir = [cdir for cdir in adir if os.path.isdir(cdir)]
            if len(adir) == 0:
                if not dirname is None:
                    adir = os.path.join(topdir[0], dirname)
                else:
                    adir = topdir[0]
                os.makedirs(adir)
                adir = iter(adir,)
            else:
                adir = iter(adir)

        adir = list(adir)
        subpath = ""
        cont = True
        if cont and not subject is None:
            subpath = os.path.join(subpath, subject)
        else:
            cont = False

        if cont and not date is None:
            subpath = os.path.join(subpath, date)
        else:
            cont = False

        if cont and not view is None:
            subpath = os.path.join(subpath, view)
        else:
            cont = False

        if create:
            adir = [os.path.join(cdir, subpath) for cdir in adir]
            edir = iter(adir)
            adir = (cdir for cdir in adir if os.path.isdir(cdir))
            try:
                adir = next(adir)
            except StopIteration:
                adir = next(edir)
                os.makedirs(adir)

            adir = iter((adir,))
        else:
            adir = (os.path.join(cdir, subpath) for cdir in adir)
            if test:
                adir = (cdir for cdir in adir if os.path.exists(cdir))

        return adir

    def getallanalysisdir(
        self,
        local=True,
        subject=None,
        date=None,
        view=None,
        create=False,
        test=True,
        anydir=False,
    ):
        return self.getalldir(
            self.analysis,
            local,
            subject,
            date,
            view,
            create=create,
            test=test,
            anydir=anydir,
        )

    def getanalysisdir(
        self, local=True, subject=None, date=None, view=None, create=False, test=True
    ):
        return next(self.getallanalysisdir(local, subject, date, view, create, test))

    def getanalysisfile(
        self,
        path,
        subject=None,
        date=None,
        view=None,
        local=True,
        test=True,
        create=True,
    ):
        filepath = os.path.join(
            self.getanalysisdir(local, subject, date, view, test=test), path
        )
        filedir = os.path.dirname(filepath)
        if create and not os.path.isdir(filedir):
            os.makedirs(filedir)

        return filepath

    def getsyncfile(self):
        return next(self.filesinpath(self.gettopdirs(), "sync.csv"))

    def getexperiments(
        self, hasdown=True, hasup=True, hasvideo=False, hassensors=True, hassync=True
    ):
        alldirs = itertools.chain.from_iterable(
            (
                glob.glob(os.path.join(topdir, "Video", "*", "*"))
                for topdir in self.gettopdirs()
            )
        )
        allexps = list(
            set(
                (os.path.basename(os.path.dirname(expdir)), os.path.basename(expdir))
                for expdir in alldirs
            )
        )

        selection = np.ones(len(allexps), dtype=bool)

        if hasdown or hasvideo:
            downexists = [
                any(
                    self.getalldir(
                        "Video",
                        False,
                        exp[0],
                        exp[1],
                        "Down",
                        create=False,
                        test=True,
                        anydir=True,
                    )
                )
                for exp in allexps
            ]
            downexists = np.array(downexists, dtype=bool)

        if hasdown:
            selection *= downexists

        if hasup or hasvideo:
            upexists = [
                any(
                    self.getalldir(
                        "Video",
                        False,
                        exp[0],
                        exp[1],
                        "Up",
                        create=False,
                        test=True,
                        anydir=True,
                    )
                )
                for exp in allexps
            ]
            upexists = np.array(upexists, dtype=bool)

        if hasup:
            selection *= upexists

        if hasvideo:
            selection *= downexists + upexists

        if hassensors:
            exists = [
                any(
                    self.getalldir(
                        "Video",
                        False,
                        exp[0],
                        exp[1],
                        "WL",
                        create=False,
                        test=True,
                        anydir=True,
                    )
                )
                for exp in allexps
            ]
            exists = np.array(exists, dtype=bool)
            selection *= exists

        if hassync:
            synctab = pd.read_csv(self.getsyncfile())
            matched = [
                len(synctab[(synctab["rat"] == exp[0]) & (synctab["date"] == exp[1])])
                > 0
                for exp in allexps
            ]
            selection *= np.array(matched, dtype=bool)

        return [exp for exp, select in zip(allexps, selection) if select]


def create_grid_video(
    inputs,
    duration,
    outfile=None,
    nrows=None,
    ncols=None,
    speed=None,
    fps="pal",
    quiet=True,
):
    """
    Create a video with a sequence of grids of equally sized views. Text is
    embedded on each view with the filename (basename) and the start position.
    Video is reencoded in new fps.

    Parameters:
    inputs - An iterable list of tuples. ALl tuples must have the same number of
             elements  (2/3).
             2 elements format: (filename, start)
             3 elements format: (filename, start, (left, top, right, bottom))
             The left, top, right, bottom define a crop box.

             A numpy array with 2 or 3 dimensions (nscense x nrows x ncols) can
             also be used, in which case the number of scenes, rows, and columns
             will be deduced from its dimensions.
    duration - Duration of each scene (common to all views).
    outfile - The output file name (None will use a generated name under /tmp).
    nrows - Number of rows (default 3)
    ncols - Number of columns (default 3)
    speed - Adjust video speed (<1 - slow motion, >1 - faster motion).
    fps - Frames per second in output video (default is pal).
    quiet - Whether to display extra messages.
    """

    #
    # Determine grid dimensions
    #
    if isinstance(inputs, np.ndarray):
        dims = inputs.shape
        if len(dims) > 3:
            raise Exception(
                f"Too many dimensions {dims}. Maximum positions dimensions is 3 (scenes x rows x cols)."
            )

        dims = [1, 1, 1] + dims
        nscenes, nrows, ncols = dims[-3:]
    else:
        nrows = 3 if nrows is None else nrows
        ncols = 3 if ncols is None else ncols
        nscenes = 1
        if len(inputs) > (nrows * ncols):
            nscenes = (len(inputs) + (nrows * ncols - 1)) // (nrows * ncols)
        elif len(inputs) > ncols:
            nrows = (len(inputs) + ncols - 1) // ncols
        else:
            ncols = len(inputs)
            nrows = 1

    #
    # Get video dimensions
    #
    probe = ffmpeg.probe(inputs[0][0])
    vidinfo = next(
        stream for stream in probe["streams"] if stream["codec_type"] == "video"
    )
    width, height = vidinfo["width"], vidinfo["height"]

    outwidth = int(width / ncols / 2) * 2
    outheight = int(height / ncols / 2) * 2

    #
    # Create input sources for ffmpeg.
    #
    clips = [ffmpeg.input(inclip[0], ss=inclip[1], t=duration) for inclip in inputs]

    #
    # Add a speed filter if required.
    #
    if not (speed is None):
        ptsmod = 1.0 / speed
        clips = [clip.filter_("setpts", f"{ptsmod}*PTS") for clip in clips]

    #
    # Compute crop box and create crop filter.
    #
    if len(inputs[0]) >= 3:
        x1 = np.array([inclip[2][0] for inclip in inputs])
        y1 = np.array([inclip[2][1] for inclip in inputs])
        x2 = np.array([inclip[2][2] for inclip in inputs])
        y2 = np.array([inclip[2][3] for inclip in inputs])
        x1 = np.maximum(x1, 0)
        x2 = np.minimum(x2, width)
        y1 = np.maximum(y1, 0)
        y2 = np.minimum(y2, height)
        w = x2 - x1
        h = y2 - y1
        r = w / h
        ratio1 = width / height
        ratio2 = height / width
        wbound = r > ratio1
        hbound = ~wbound
        h[wbound] = w[wbound] * ratio2
        w[hbound] = h[hbound] * ratio1
        clips = [
            clip.filter_("crop", x=cx, y=cy, w=cw, h=ch)
            for clip, cx, cy, cw, ch in zip(clips, x1, y1, w, h)
        ]

    #
    # Extra filters:
    # Reencode fps.
    # Scale to size of grid boxes.
    # Draw text on each grid box view.
    # #
    fontsize = int(30 * outheight / height)
    box_height = int(outheight / fontsize * 1.01)

    clips = [
        clip.filter_("fps", fps)
        .filter_(
            "scale", w=outwidth, h=outheight, force_original_aspect_ratio="disable"
        )
        .filter_("setsar", r="1/1")
        .drawbox(
            x=0, y=0, width=outwidth, height=box_height, color="Black@0.8", thickness=10
        )
        .drawtext(
            text=f'{inclip [0].split ("/") [-1]}/{inclip [1]:.2f}',
            fontcolor="Cyan",
            fontsize=fontsize,
            x=1,
            y=1,
        )
        for clip, inclip in zip(clips, inputs)
    ]

    #
    # Add placeholders if number of available grid boxes is greater than
    # available clips. Text is added as a workaround to a bug in the package.
    #
    nzones = nscenes * nrows * ncols
    if len(clips) < nzones:
        clips += [
            ffmpeg.input(
                f"color=size={outwidth}x{outheight}:"
                + f"color=Gray:rate=pal:d={duration}",
                f="lavfi",
            ).drawtext(
                text=f"PlaceHolder {i}", fontcolor="White", fontsize=32, x=20, y=20
            )
            for i in range(nzones - len(clips))
        ]

    pos = 0
    if outfile is None:
        outfile = tempfile.NamedTemporaryFile(suffix=".mp4").name

    #
    # Create temporary file for each scene (a scene is a single grid).
    #
    tfiles = []
    for sceneno in range(nscenes):
        rows = []
        for _ in range(nrows):
            #
            # Stack boxes in a row
            #
            cols = [clips[pos + col] for col in range(ncols)]
            if len(cols) == 1:
                rows.append(cols[0])
            else:
                rows.append(
                    ffmpeg.filter_multi_output(cols, "hstack", inputs=ncols).stream()
                )
            pos += len(cols)

        #
        # Stack rows in a scene.
        #
        if len(rows) == 1:
            scene = rows[0]
        else:
            scene = ffmpeg.filter_multi_output(rows, "vstack", inputs=nrows).stream()

        #
        # Start the process to create the scene.
        #
        tfile = tempfile.NamedTemporaryFile(suffix=".mp4").name
        if not quiet:
            logging.info(scene.output(tfile).compile())
        scene.output(tfile).run(quiet=quiet)
        tfiles.append(tfile)
        logging.info(f"Scene {sceneno} done ({(sceneno+1)*100/nscenes:.0f}%).")

    if not quiet:
        logging.info(
            ffmpeg.concat(*[ffmpeg.input(tfile) for tfile in tfiles])
            .output(outfile, y=None)
            .compile()
        )

    #
    # Concatenate the scenese into a single video. This is in fact double work,
    # as the concatenation could be done as a final filter, but I preferred the
    # extra work (and time), to ease problems debugging each scene creation.
    #
    logging.info(f"Concatenating {len (tfiles)} scenes.")
    (
        ffmpeg.concat(*[ffmpeg.input(tfile) for tfile in tfiles])
        .output(outfile, y=None)
        .run(quiet=quiet)
    )

    #
    # Remove the temporary files.
    #
    [os.remove(tfile) for tfile in tfiles]

    return outfile
