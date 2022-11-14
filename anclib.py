import phylotreelib as pt
import sequencelib
import re
import math
import collections
import numpy as np

###################################################################################################
###################################################################################################

class AnclibError(Exception):
    pass

###################################################################################################
###################################################################################################

class Anc_recon():
    """Objects of this class contain a tree and a corresponding set of sequences. 
    
    Some sequences correspond to leaves on the tree. These are observed sequences.
    Other sequences correspond to internal nodes on the tree. These are ancestral reconstructions.
    
    There is also information about credibility of sequence states: 100% for leaves, possibly
    less for internal nodes. If there is no information about credibility, then these are
    set to 100%.
    
    Objects may also contain information about other, non-sequence, traits. These also have 
    associated credibilities.
    
    There are several constructors corresponding to different sources of information.
    There are also methods for adding extra information to already constructed objects.
    """

    def __init__(self):
        self.tree = None
        self.alignment = None
        self.seqprobs = None
        self.traitdict = None
        self.traitprobs = None
        self.nodeid2seqname = None
        self.seqname2nodeid = None
        self.i2seqname = None
        self.seqname2i = None
        
    ###############################################################################################

    def add_from_basemlrst(self, filename):
        """Add information from BASEML rst file: 
            tree
            sequences (including ancestral reconstructions)
            probabilities for seq residues (2D numpy array)
            dicts mappings between nodeid (from tree) and seqname
            dicts mapping between array row index and seqname
        """
        
        bf = Baseml_rstfile(filename)
        self.tree = bf.read_tree()
        self.alignment, self.seqprobs = bf.read_alignment()
        self.nodeid2seqname, self.seqname2nodeid = bf.nodeid2seqname, bf.seqname2nodeid
        self.i2seqname, self.seqname2i = bf.i2seqname, bf.seqname2i
        
    ###############################################################################################

    def add_from_mbasr(self,filename):
        pass
        

###################################################################################################
###################################################################################################

class Baseml_rstfile():
    """Class representing parser for BASEML rst file. 
    
    Methods for extracing tree, alignment, and residue probabilities for ancestral 
    (and contemporaneous) sequences.
    
    Tree is constructed based on branch info (87..88 87..89 etc.) and newick string.
    Internal nodes are numbered according to branch info. Leaves are named from Newick string.
    
    Ancestral sequences (corresponding to internal nodes on tree) are named based on nodeIDs.
    The method also returns
    """

    ###############################################################################################

    def __init__(self, filename=None, filecontent=None):
        """Check file contains needed info about tree and sequences.
        Construct dictionaries mapping between nodeID, seqname, and index"""

        num_args = (filename is not None) + (filecontent is not None)
        if num_args != 1:
            raise AnclibError("Baseml_rstfile __init__ requires either filename or filecontent (not both)")
        elif filecontent:
            self.rstfile = StringIO(filecontent)
        else:
            self.rstfile = open(filename, mode="rt", encoding="UTF-8")
        
        self._read_until("^Ancestral reconstruction by BASEML")
        self.rstfile.readline()
        self.newicknames = self.rstfile.readline()

        self.rstfile.readline()
        self.newicknumbers = self.rstfile.readline()

        # Find line with pattern like "   88..89  88..90"
        tmp,_ = self._read_until("^\s*[0-9]+\.\.[0-9]+\s*[0-9]+\.\.[0-9]+", nsave=1)
        self.branchlist = tmp[0].strip().split()
        
        tmp,_ = self._read_until("^Nodes [0-9]+ to [0-9]+ are ancestral", nsave=1)
        self.nseq = int(tmp[0].split()[3])
        
        # Find first line of seqinfo. Pattern like "   1     86   AAAAAAAAAAAAA"
        _,self.startofseqpos = self._read_until("^\s*1\s+[0-9]+\s+[A-Z]+")
        tmp,_ = self._read_until("^\n",nsave=2)
        self.seqlen = int(tmp[0].split()[0])
        
        #### NEED TO FIND SEQTYPE!!!!!
        
        # Note: nodenums start at 1, not 0 (this is BASEML node-numbering scheme)
        self.nodeid2nodenum, self.nodenum2nodeid = self._map_nodeid_nodenum()
        self.nodeid2seqname, self.seqname2nodeid = self._map_nodeid_name()
        self.nodenum2seqname, self.seqname2nodenum = self._map_nodenum_name()
        self.i2seqname, self.seqname2i = self._map_i_seqname()
        self.seqnames = set(self.seqname2nodeid.keys())
                
    ###########################################################################################

    def _map_nodeid_nodenum(self):
        
        # Assumes the two newick strings in rst are in the same order
        leaflist = re.sub(",|;|\(|\)|: [0-9]\.[0-9]+", "", self.newicknames).split()
        numtmp = re.sub(",|;|\(|\)", "", self.newicknumbers).split()
        numberlist = [int(x) for x in numtmp]
        nodenum2nodeid = dict(zip(numberlist, leaflist))
        for nodenum in range(max(numberlist) + 1, self.nseq + 1):
            nodenum2nodeid[nodenum] = nodenum
        nodeid2nodenum = {val:key for key,val in nodenum2nodeid.items()}
        return (nodeid2nodenum, nodenum2nodeid)

    ###########################################################################################
    
    def _map_nodeid_name(self):
        nodeid2seqname = {} 
        seqname2nodeid = {}
        ndigits = math.ceil(math.log10(self.nseq))
        for nodenum,nodeid in self.nodenum2nodeid.items():
            if nodeid == nodenum:
                name = "seq_{1:0{0}}".format(ndigits, nodeid)
            else:
                name = nodeid
            nodeid2seqname[nodeid] = name
            seqname2nodeid[name] = nodeid
        return (nodeid2seqname, seqname2nodeid)

    ###########################################################################################
    
    def _map_nodenum_name(self):
        nodenum2seqname = {} 
        seqname2nodenum = {}
        for nodenum,nodeid in self.nodenum2nodeid.items():
            seqname = self.nodeid2seqname[nodeid]
            nodenum2seqname[nodenum] = seqname
            seqname2nodenum[seqname] = nodenum
        return (nodenum2seqname, seqname2nodenum)

    ###########################################################################################

    def _map_i_seqname(self):
        i2seqname = {}
        seqname2i = {}
        for nodenum,seqname in self.nodenum2seqname.items():
            i2seqname[nodenum - 1] = seqname
            seqname2i[seqname] = nodenum - 1
        return (i2seqname, seqname2i)
        
    ###########################################################################################

    def read_tree(self):
        """Extracts tree from branch info and Newick string. Returns phylotreelib.Tree object"""
        
        parentlist, childlist = self._parsebranches()
        return pt.Tree.from_branchinfo(parentlist, childlist)
        
    ###########################################################################################

    def _parsebranches(self):
        parentlist = []
        childlist = []
        for branch in self.branchlist:
            parent,child = branch.split("..")
            parent = self.nodenum2nodeid[int(parent)]
            child = self.nodenum2nodeid[int(child)]
            parentlist.append(parent)
            childlist.append(child)            
        return(parentlist, childlist)
        
    ###########################################################################################

    def read_alignment(self):
        """Extracts sequence information corresponding to extant and ancestral sequences.
        Returns sequencelib.Seq_alignment object and 2D numpy array with residue probs"""
        
        # Initialise lists of nseq empty lists to keep sequence and prob info
        seqlists = [ [] for i in range(self.nseq)]
        seqprobs = np.zeros((self.nseq, self.seqlen))
        
        # Move filepointer to first line of sequence info
        self.rstfile.seek(self.startofseqpos)
        
        # Read seqinfo one line (=site) at a time, append residues and probs to relevant lists
        for line in self.rstfile:
            if line == "\n":
                break
            site,residues,probs = self._parse_siteline(line)
            for i,res in enumerate(residues):
                seqlists[i].append(res)
            for i,prob in enumerate(probs):
                seqprobs[i,site-1] = prob
                
        # Determine seqtype from first seq (assume it is representative)
        seqtype = sequencelib.find_seqtype(seqlists[0])
        if seqtype not in ["DNA", "protein"]:
            raise AnclibError("Expecting either DNA or protein seqtype. Actual seqtype: {}".format(seqtype))
                
        # Create alignment from lists of residues in seqs
        alignment = sequencelib.Seq_alignment(seqtype=seqtype)
        for i in range(self.nseq):
            seq = "".join(seqlists[i])
            name = self.i2seqname[i]
            if seqtype == "DNA":
                seqobject = sequencelib.DNA_sequence(name, seq)
            else:
                seqobject = sequencelib.Protein_sequence(name, seq)
            alignment.addseq(seqobject)
        return (alignment, seqprobs)

    ###########################################################################################

    def _parse_siteline(self, line):
        words = line.split()
        site = int(words[0])
        leafres = words[2].replace(":","")
        residues = [residue for residue in leafres]
        probs = [1.00] * len(leafres)
        for word in words[3:]:
            word = word.replace(")","")
            residue,prob = word.split("(")
            prob = float(prob)
            residues.append(residue)
            probs.append(prob)
        return site,residues,probs

    ###########################################################################################

    def _read_until(self, regex, nsave=0):
        """Read filecontent up to and including regular expression pattern.
        Return last nsave lines if requested.
        Also returns position for start of last line (so can rewind one line using seek)"""

        lastnlines = collections.deque([None]*nsave)
        while True:
            prevlinepos = self.rstfile.tell()
            line = self.rstfile.readline()
            if line == "":
                msg = "BASEML rst file format not as expected"
                raise AnclibError(msg)
            if nsave > 0:
                lastnlines.append(line)
                lastnlines.popleft()
            if re.search(regex, line):
                return (lastnlines, prevlinepos)

###################################################################################################
###################################################################################################

class MBASR_file():
    """Class representing parser for MBASR ancestral reconstruction file. 
    
    Methods for extracing one tree and one alignment of ancestral (and contemporaneous) sequences.
    
    Tree is constructed based on branch info (87..88 87..89 etc.) and newick string.
    Internal nodes are numbered according to branch info. Leaves are named from Newick string.
    
    Ancestral sequences (corresponding to internal nodes on tree) are named based on nodeIDs.
    The method also returns
    """
###################################################################################################
###################################################################################################


###################################################################################################
###################################################################################################

