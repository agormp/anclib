import collections
import math
import re

import sequencelib
import phylotreelib as pt

import numpy as np
import pandas as pd
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

###################################################################################################
###################################################################################################

class AncError(Exception):
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
        self.trait = {}
        self.traitprobs = {}
        self.nodes = None
        self.sortednodes = None
        
    ###############################################################################################

    @classmethod
    def from_baseml_mbasr(cls, baseml_rstfile, 
                              mbasr_treefile, mbasr_intnodefile, mbasr_leafstatefile,
                              state0="state_0", state1="state_1"):
        """Combine information from BASEML rst file, and MBASR ancestral state reconstruction
        
        The following attributes are added:
            tree
            alignment of sequences, including ancestral reconstructions. Seqname = str(nodeid)
            probabilities for seq residues (dict of nodeid:numpy array)
            trait state (dict of nodeid:state)
            trait state probabillity (dict of nodeid:state probability)
                              
        The nodeids used by MBASR are changed to match those used by BASEML
        """
        
        obj = cls()
        ba_file = _Baseml_rstfile(baseml_rstfile)
        obj.tree = ba_file.read_tree()
        obj.alignment, obj.seqprobs = ba_file.read_alignment()
        
        mb_file = _MBASR_file(mbasr_treefile, mbasr_intnodefile, mbasr_leafstatefile, state0, state1)
        mb_tree = mb_file.read_tree()
        mb_trait, mb_probs = mb_file.read_trait()
        obj.nodes = mb_tree.nodes
        obj.sortedintnodes = sorted(list(mb_tree.intnodes))
        obj.sortednodes = sorted(list(mb_tree.leaves))
        obj.sortednodes.extend(sorted(list(mb_tree.intnodes)))
        
        mbnode2banode, unmatch_baseml, unmatch_mbasr = mb_tree.match_intnodes(obj.tree)
        if (unmatch_baseml != None) or (unmatch_mbasr != None):
            raise AncError("BASEML and MBASR trees are rooted differently. Cannot merge info")
        for leafnode in obj.tree.leaves:
            mbnode2banode[leafnode] = leafnode      # Useful in for-loop below
        for mbnode,banode in mbnode2banode.items():
            obj.trait[banode] = mb_trait[mbnode]
            obj.traitprobs[banode] = mb_probs[mbnode]
        return obj
        
    ###############################################################################################
    
    def nodeinfo(self, varseq=False, poslist=None, zeroindex=True):
        """Outputs one line per node with following informations: 
            nodeid  traitstate  seqstates
        option varseq=True (default) outputs only variable sites from sequences.
        Can also explicitly provide poslist of sites to be printed. Indexing in poslist
        can start at 0 (zeroindex=True) or 1 (zeroindex=False)"""
        
        pos = None
        if varseq and poslist:
            raise AncError("Specify either varseq or poslist option - not both")
        if varseq:
            pos = self.alignment.varcols()
        elif poslist:
            pos = poslist
            
        if pos and not zeroindex:
            pos = [p - 1 for p in pos]
            
        for nodeid in self.sortednodes:
            seqname = str(nodeid)
            trait = self.trait[nodeid]
            if pos:
                seq = self.alignment.getseq(seqname).subseqpos(pos).seq
            else:
                seq = self.alignment.getseq(seqname).seq
            print("{}\t{}\t{}".format(nodeid, trait, seq))
    
    ###############################################################################################
    
    def branchinfo(self, varseq=False, poslist=None, zeroindex=True, 
                    printif_traitdiff=False, printif_seqdiff=False):
        """Outputs one line per branch with following informations: 
            nodeid_from   nodeid_to   trait_from   trait_to   seq_from   seq_to
        Option varseq=True (default) output only variable sites from sequences.
        Option poslist: specify sites to be printed. 
        Option zeroindex=True: Start indexing of poslist at 0 (otherwise start at 1)
        Option printif_traitdiff=True: only print branches where traits differ
        Option printif_seqdiff=True: only print branches where selected residues have changed"""
        
        pos = None
        if varseq and poslist:
            raise AncError("Specify either varseq or poslist option - not both")
        if varseq:
            pos = self.alignment.varcols()
        elif poslist:
            pos = poslist
    
        if pos and not zeroindex:
            pos = [p - 1 for p in pos]
            
        for nodefrom in self.sortedintnodes:
            for nodeto in self.tree.children(nodefrom):     
                seqnamefrom = str(nodefrom)
                seqnameto = str(nodeto)
                traitfrom = self.trait[nodefrom]
                traitto = self.trait[nodeto]
                if pos:
                    seqfrom = self.alignment.getseq(seqnamefrom).subseqpos(pos).seq
                    seqto = self.alignment.getseq(seqnameto).subseqpos(pos).seq
                else:
                    seqfrom = self.alignment.getseq(seqnamefrom).seq
                    seqto = self.alignment.getseq(seqnameto).seq

                # Could test for traitdiff before setting seq, but neater code this way...
                printbranch = True       
                if printif_traitdiff and (traitfrom==traitto):
                    printbranch = False
                if printif_seqdiff and (seqfrom==seqto):
                    printbranch = False
                if printbranch:
                    print("{}\t{}\t{}\t{}\t{}\t{}".format(
                           nodefrom, nodeto, traitfrom, traitto, seqfrom, seqto))

    ###############################################################################################
    
    def branchdiff(self, zeroindex=True, printif_traitdiff=False):
        """Prints one line of output for each sequence change on all branches of the tree:
            node_from   node_to   site  trait_from   trait_to   residue_from   residue_to
        Here 'site' is the index of the sequence residue.
        Option zeroindex=False causes numbering to start at 1 (otherwise at 0)
        Option printif_traitdiff=True: only print branches where traits differ
        """
        
        for nodefrom in self.sortedintnodes:
            for nodeto in self.tree.children(nodefrom):     
                seqnamefrom = str(nodefrom)
                seqnameto = str(nodeto)
                traitfrom = self.trait[nodefrom]
                traitto = self.trait[nodeto]
                seqfrom = self.alignment.getseq(seqnamefrom)
                seqto = self.alignment.getseq(seqnameto)
                difflist = seqfrom.seqdiff(seqto, zeroindex)

                # Could test for traitdiff before setting seq, but neater code this way...
                if (not printif_traitdiff) or (traitfrom!=traitto):
                    for site,resfrom,resto in difflist:
                        print("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                           nodefrom, nodeto, site, traitfrom, traitto, resfrom, resto))
        
    ###############################################################################################
    
    def varpos(self, zeroindex=True):
        """Returns list of variable (unconserved) sites in alignment of sequences"""
        
        varpos = self.alignment.varcols()
        if not zeroindex:
            varpos = [pos+1 for pos in varpos]
        return varpos

###################################################################################################
###################################################################################################

class _Baseml_rstfile():
    """Class representing parser for BASEML rst file. 
    
    Methods for extracing tree, alignment, and residue probabilities for ancestral 
    (and contemporaneous) sequences.
    
    Tree is constructed based on branch info (87..88 87..89 etc.) and newick string.
    Internal nodes are numbered according to branch info. Leaves are named from Newick string.
    
    Ancestral sequences (corresponding to internal nodes on tree) are named based on nodeIDs.
    The method also returns
    """

    ###############################################################################################

    def __init__(self, filename=None):
        """Check file contains needed info about tree and sequences.
        Construct dictionaries mapping between nodeID, seqname, and index"""

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
        # The find last line of seqinfo to get seqlen
        _,self.startofseqpos = self._read_until("^\s*1\s+[0-9]+\s+[A-Z]+")
        tmp,_ = self._read_until("^\n",nsave=2)
        self.seqlen = int(tmp[0].split()[0])
                
        # Note: nodenums start at 1, not 0 (this is BASEML node-numbering scheme)
        self.nodeid2nodenum, self.nodenum2nodeid = self._map_nodeid_nodenum()
        self.nodeids = set(self.nodeid2nodenum.keys())
                
    ###########################################################################################

    def close(self):
        self.rstfile.close()

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
        
        seqlists = [ [] for i in range(self.nseq)]
        seqprobs = {nodeid:np.zeros(self.seqlen) for nodeid in self.nodeids}
        
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
                nodenum = i + 1
                nodeid = self.nodenum2nodeid[nodenum]
                seqprobs[nodeid][site-1] = prob
                
        # Determine seqtype from first seq (assume it is representative)
        seqtype = sequencelib.find_seqtype(seqlists[0])
        if seqtype not in ["DNA", "protein"]:
            raise AncError("Expecting either DNA or protein seqtype. Actual seqtype: {}".format(seqtype))
                
        # Create alignment from lists of residues in seqs
        alignment = sequencelib.Seq_alignment(seqtype=seqtype)
        for i in range(self.nseq):
            seq = "".join(seqlists[i])
            nodenum = i + 1
            nodeid = self.nodenum2nodeid[nodenum]
            name = str(nodeid)      # Leaves are already strings, so no effect on those
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
                raise AncError(msg)
            if nsave > 0:
                lastnlines.append(line)
                lastnlines.popleft()
            if re.search(regex, line):
                return (lastnlines, prevlinepos)

###################################################################################################
###################################################################################################

class _MBASR_file():
    """Class representing parser for MBASR ancestral reconstruction file. 
    
    Methods for extracing one tree and one alignment of ancestral (and contemporaneous) sequences.
    
    Tree is constructed based on branch info (87..88 87..89 etc.) and newick string.
    Internal nodes are numbered according to branch info. Leaves are named from Newick string.
    
    Ancestral sequences (corresponding to internal nodes on tree) are named based on nodeIDs.
    The method also returns
    """
    
    def __init__(self, treefile, intnodefile, leafstatefile, state0, state1):
        """Read files, store as dataframes, ready for further parsing.
        Tree is read as dataframe (so we have info about internal nodeIDs):
            parent, node, branch.length,	label
        Intnode reconstructions are read as dataframe:
            nodeid, pstate1, pstate2
        Leafnode states read as dataframe:
            leafnodeid, state
        """ 
        
        # Read treefile, store branch info as pandas dataframe
        # Note: I am cargo culting the rpy2 code here...
        # Format: 
        ape = importr("ape")
        tidytree = importr("tidytree")    
        tree = ape.read_tree(treefile)
        dftreeR = tidytree.as_tibble_phylo(tree)
        with localconverter(robjects.default_converter + pandas2ri.converter):
            self.dftreePy = robjects.conversion.rpy2py(dftreeR)
        self.nseq = self.dftreePy.shape[0]
        
        # Read internal node ancestral reconstruction state info
        self.intnodestate = pd.read_csv(intnodefile, delim_whitespace=True,
                                        header=None, names=["nodeID","pstate0","pstate1"])
        self.nintnode = self.intnodestate.shape[0]
        
        # Read leaf node state info
        self.leafnodestate = pd.read_csv(leafstatefile, delim_whitespace=True,
                                        header=None, names=["leafID","state"])
        self.nleaf = self.leafnodestate.shape[0]
        self.state0 = state0
        self.state1 = state1
        
    ###########################################################################################

    def read_tree(self):
        parentlist = []
        childlist = []
        
        for i in range(self.nseq):
            parent, node, label = self.dftreePy.iloc[i,[0,1,3]]
            # Tidy tree df has one row where root is both parent and node: Discard.
            if parent != node:
                parentlist.append(parent)
                # Tidytree parent is either leafname (type: string) or 
                # NA_character_ (type: rpy2.rinterface_lib.sexp.NACharacterType)
                if type(label) == str:
                    childlist.append(label)
                else:
                    childlist.append(node)
        tree = pt.Tree.from_branchinfo(parentlist, childlist)
        return tree

    ###########################################################################################

    def read_trait(self):
        statenamedict = {0:self.state0, 1:self.state1}
        nodeidlist = list(self.leafnodestate["leafID"])
        intnodeidlist = [int(x.replace("node","")) for x in self.intnodestate["nodeID"]]
        nodeidlist.extend(intnodeidlist)
        
        traitstatelist = [statenamedict[state] for state in self.leafnodestate["state"]]
        traitproblist = [1.0] * self.nleaf
        intnodestatelist = []
        intproblist = []
        for i in range(self.nintnode):
            p0, p1 = self.intnodestate.iloc[i,[1,2]]
            if p0 > 0.5:
                intnodestatelist.append(self.state0)
                intproblist.append(p0)
            else:
                intnodestatelist.append(self.state1)
                intproblist.append(p1)
        traitstatelist.extend(intnodestatelist)
        traitproblist.extend(intproblist)
        
        traitdict = dict(zip(nodeidlist, traitstatelist))
        traitprobdict = dict(zip(nodeidlist, traitproblist))
        
        return (traitdict, traitprobdict)

###################################################################################################
###################################################################################################


