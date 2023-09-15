import collections
import csv
import math
import re

import sequencelib
import phylotreelib as pt

import numpy as np
import pandas as pd

###################################################################################################
###################################################################################################

class AncError(Exception):
    pass

###################################################################################################
###################################################################################################

class AncRecon:
    """Created from one ancseq object, and one anctrait object.
    Contains info about tree, and also about sequences and traits (ancestral and reconstructed)

    Some seqs/trait correspond to leaves on the tree. These are observed seqs/trait.
    Other seqs/trait correspond to internal nodes. These are ancestral reconstructions.

    There may also be information about credibility of reconstructed seqs/traits
    (100% for leaves, possibly less for internal nodes)

    Object can output information about nodes, branches, changes, etc.

    Internal node IDs are set to be the same as in the anctrait object

    NOTE: in state of transition. Currently does not get seqprob
    """

    def __init__(self, ancseq, anctrait):
        self.tree = anctrait.tree
        self.traitdict = anctrait.traitdict
        self.traitprob = anctrait.traitprob    # Python note: Always there?

        self.sortedintnodes = sorted(list(self.tree.intnodes))
        self.sortednodes = sorted(list(self.tree.leaves))
        self.sortednodes.extend(self.sortedintnodes)

        seqid2traitid, u1, u2 = ancseq.tree.match_nodes(self.tree)
        if (u1 != None) or (u2 != None):
            raise AncError("Seq-tree and trait-tree are rooted differently. Cannot merge info")

        self.alignment = sequencelib.Seq_alignment(seqtype=ancseq.alignment.seqtype)
        for seq in ancseq.alignment:
            seqid = ancseq.name2id[seq.name]  # Python note: not this objects responsibility!!!
            newname = str(seqid2traitid[seqid])
            seq.rename(newname)
            self.alignment.addseq(seq)

    ###############################################################################################

    def write_tree(self, outfilename, label="branchtype", br=1):
        """Prints Nexus tree to outfilename, where branch labels carry information:

        label="branchtype": labels give same inforamtion as branchtype in branchinfo and
        branchdiff, namely traits for parent, nodefrom, nodeto (e.g., human-swine-swine)

        label="nodeid": nodeid used in other output functions (This means labels should
        be interpreted as belonging to the child node, not the branch)"""

        out_tree = self.tree.copy_treeobject(copylabels=False)

        if label == "branchtype":
            for nodefrom in out_tree.intnodes:
                for nodeto in out_tree.children(nodefrom):
                    traitfrom = self.traitdict[nodefrom]
                    traitto = self.traitdict[nodeto]
                    if br==1:
                        branch_type = f"{traitfrom}-{traitto}"
                    elif br==2:
                        if nodefrom != self.tree.root:
                            parent = self.tree.parent(nodefrom)
                            traitfrom_parent = self.traitdict[parent]
                        else:
                            traitfrom_parent = "None"
                        branch_type = f"{traitfrom_parent}-{traitfrom}-{traitto}"
                    out_tree.setlabel(nodefrom, nodeto, branch_type)
            treestring = out_tree.nexus()
        elif label == "nodeid":
            # Hack: add label for root manually
            out_tree.set_nodeid_labels()
            treestring = out_tree.nexus()
            rootid = str(out_tree.root)
            treestring = treestring.replace(");", f"){rootid};")
        else:
            raise AncError("'label' must be 'branchtype' or 'nodeid'")

        print(out_tree) #DEBUG

        with open(outfilename, "w") as outfile:
            outfile.write(treestring)

    ###############################################################################################

    def nodeinfo(self, outfilename, varseq=False, poslist=None, zeroindex=True, probmin=None,
                 translate=False):
        """Writes results to 'outfilename':
        Output is one line per node in tree - with following informations:
            nodeid  traitstate  traitprob seqstates
        option varseq=True outputs only variable sites from sequences.
        option probmin (if not None): Only print info for nodes with traitprob > probmin
        option translate: Translate DNA to amino acid sequences. Note position info is now for aa
        Can also explicitly provide poslist of sites to be printed. Indexing in poslist
        can start at 0 (zeroindex=True) or 1 (zeroindex=False)"""

        with open(outfilename, "w") as outfile:
            outfile.write("# {}\t{}\t{}\t{}\n".format("nodeid", "trait", "traitprob", "seq"))
            pos = None
            if varseq and poslist:
                raise AncError("Specify either varseq or poslist option - not both")
            if translate:
                alignment = self.alignment.translate()
            else:
                alignment = self.alignment
            if varseq:
                pos = alignment.varcols()
            elif poslist:
                pos = poslist

            if pos and not zeroindex:
                pos = [p - 1 for p in pos]

            for nodeid in self.sortednodes:
                if (not probmin) or (self.traitprob[nodeid] > probmin):
                    seqname = str(nodeid)
                    trait = self.traitdict[nodeid]
                    traitprob = self.traitprob[nodeid]
                    if pos:
                        seq = alignment.getseq(seqname).subseqpos(pos).seq
                    else:
                        seq = alignment.getseq(seqname).seq
                    outfile.write(f"{nodeid}\t{trait}\t{traitprob:.3f}\t{seq}\n")

    ###############################################################################################

    def branchinfo(self, outfilename, varseq=False, poslist=None, zeroindex=True,
                    printif_traitdiff=False, printif_seqdiff=False, probmin=None,
                    translate=False, br=1):
        """Writes results to 'outfilename':
        Outputs one line per branch with following informations:
          nodeid_from  nodeid_to  branchlen trait_from  trait_to
                                            traitprob_from  traitprob_to seq_from  seq_to
                                            branchtype

        varseq=True (default) output only variable sites from sequences.
        poslist: specify sites to be printed.
        zeroindex=True: Start indexing of poslist at 0 (otherwise start at 1)
        printif_traitdiff=True: only print branches where traits differ
        printif_seqdiff=True: only print branches where selected residues have changed
        probmin (if not None): Only print info where both nodes have traitprob > probmin
        translate: Translate DNA to amino acid sequences. Note position info is now for aa
        br=1: branchtype 1st degree traits: node_from to nodeto (hu-sw)
        br=2: branchtype 2nd degree traits: parent to node_from to nodeto (hu-hu-sw)
        """

        with open(outfilename, "w") as outfile:
            outfile.write("# {}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                   "node_from", "node_to", "branch_len", "trait_from", "trait_to",
                   "traitprob_from", "traitprob_to", "seq_from", "seq_to", "branch_type"))
            pos = None
            if varseq and poslist:
                raise AncError("Specify either varseq or poslist option - not both")
            if translate:
                alignment = self.alignment.translate()
            else:
                alignment = self.alignment
            if varseq:
                pos = alignment.varcols()
            elif poslist:
                pos = poslist

            if pos and not zeroindex:
                pos = [p - 1 for p in pos]

            for nodefrom in self.sortedintnodes:
                for nodeto in self.tree.children(nodefrom):
                    if (not probmin) or (
                            (self.traitprob[nodefrom] > probmin) and
                            (self.traitprob[nodeto] > probmin)):
                        seqnamefrom = str(nodefrom)
                        seqnameto = str(nodeto)
                        blen = self.tree.nodedist(nodefrom, nodeto)
                        traitfrom = self.traitdict[nodefrom]
                        traitto = self.traitdict[nodeto]
                        if br==1:
                            branch_type = f"{traitfrom}-{traitto}"
                        elif br==2:
                            if nodefrom != self.tree.root:
                                parent = self.tree.parent(nodefrom)
                                traitfrom_parent = self.traitdict[parent]
                            else:
                                traitfrom_parent = "None"
                            branch_type = f"{traitfrom_parent}-{traitfrom}-{traitto}"
                        traitprobfrom = self.traitprob[nodefrom]
                        traitprobto = self.traitprob[nodeto]
                        if pos:
                            seqfrom = alignment.getseq(seqnamefrom).subseqpos(pos).seq
                            seqto = alignment.getseq(seqnameto).subseqpos(pos).seq
                        else:
                            seqfrom = alignment.getseq(seqnamefrom).seq
                            seqto = alignment.getseq(seqnameto).seq

                        # Could test for traitdiff before setting seq, but neater code this way...
                        printbranch = True
                        if printif_traitdiff and (traitfrom==traitto):
                            printbranch = False
                        if printif_seqdiff and (seqfrom==seqto):
                            printbranch = False
                        if printbranch:
                            outfile.write("{}\t{}\t{}\t{}\t{}\t{:.3f}\t{:.3f}\t{}\t{}\t{}\n".format(
                                   nodefrom, nodeto, blen, traitfrom, traitto,
                                   traitprobfrom, traitprobto, seqfrom, seqto, branch_type))

    ###############################################################################################

    def branchdiff(self, outfilename, zeroindex=True, printif_traitdiff=False, probmin=None,
                   translate=False, br=1):
        """Writes results to 'outfilename':
        Output is one line of output for each sequence change, on all branches of the tree:
            node_from, node_to, branch_len, site, trait_from, trait_to,
                                      traitprob_from, traitprob_to, residue_from, residue_to,

        Here 'site' is the index of the sequence residue.
        Option zeroindex=False causes numbering to start at 1 (otherwise at 0)
        Option printif_traitdiff=True: only print branches where traits differ
        Option probmin (if not None): Only print info where both nodes have traitprob > probmin
        Option translate: Translate DNA to amino acid sequences. Note position info is now for aa
        """

        with open(outfilename, "w") as outfile:
            outfile.write("# {}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
               "node_from", "node_to", "branch_len", "seqpos", "trait_from", "trait_to",
               "traitprob_from", "traitprob_to", "residue_from", "residue_to", "branch_type"))
            if translate:
                alignment = self.alignment.translate()
            else:
                alignment = self.alignment
            for nodefrom in self.sortedintnodes:
                for nodeto in self.tree.children(nodefrom):
                    if (not probmin) or (
                            (self.traitprob[nodefrom] > probmin) and
                            (self.traitprob[nodeto] > probmin)):
                        seqnamefrom = str(nodefrom)
                        seqnameto = str(nodeto)
                        blen = self.tree.nodedist(nodefrom, nodeto)
                        traitfrom = self.traitdict[nodefrom]
                        traitto = self.traitdict[nodeto]
                        if br==1:
                            branch_type = f"{traitfrom}-{traitto}"
                        elif br==2:
                            if nodefrom != self.tree.root:
                                parent = self.tree.parent(nodefrom)
                                traitfrom_parent = self.traitdict[parent]
                            else:
                                traitfrom_parent = "None"
                            branch_type = f"{traitfrom_parent}-{traitfrom}-{traitto}"
                        traitprobfrom = self.traitprob[nodefrom]
                        traitprobto = self.traitprob[nodeto]
                        seqfrom = alignment.getseq(seqnamefrom)
                        seqto = alignment.getseq(seqnameto)
                        difflist = seqfrom.seqdiff(seqto, zeroindex)

                        # Could test for traitdiff before setting seq, but neater code this way...
                        if (not printif_traitdiff) or (traitfrom!=traitto):
                            for site,resfrom,resto in difflist:
                                outfile.write("{}\t{}\t{}\t{}\t{}\t{}\t{:.3f}\t{:.3f}\t{}\t{}\t{}\n".format(
                                   nodefrom, nodeto, blen, site, traitfrom, traitto,
                                   traitprobfrom, traitprobto, resfrom, resto, branch_type))

###################################################################################################
###################################################################################################

class BasemlSeq:
    """Class representing parser for BASEML rst file.

    Methods for extracting tree, alignment, and residue probabilities for ancestral
    (and contemporaneous) sequences.

    Tree is constructed based on branch info (87..88 87..89 etc.) and newick string.
    Internal nodes are numbered according to branch info. Leaves are named from Newick string.

    Ancestral sequences (corresponding to internal nodes on tree) are named based on nodeIDs.
    The method also returns
    """

    ###############################################################################################

    def __init__(self, rstfile):
        """Check file contains needed info about tree and sequences.
        Construct dictionaries mapping between nodeID, seqname, and index"""

        self.rstfile = open(rstfile, mode="rt", encoding="UTF-8")

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
        # Then find last line of seqinfo to get seqlen
        _,self.startofseqpos = self._read_until("^\s*1\s+[0-9]+\s+[A-Z]+")
        tmp,_ = self._read_until("^\n",nsave=2)
        self.seqlen = int(tmp[0].split()[0])

        # Note: nodenums start at 1, not 0 (this is BASEML node-numbering scheme)
        self.nodeid2nodenum, self.nodenum2nodeid = self._map_nodeid_nodenum()
        self.nodeids = set(self.nodeid2nodenum.keys())
        self.tree = self.get_tree()
        self.alignment, self.seqprob = self.get_alignment_info()

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

    def get_tree(self):
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

    def get_alignment_info(self):
        """Extracts sequence information corresponding to extant and ancestral sequences.
        Returns sequencelib.Seq_alignment object and 2D numpy array with residue probs"""

        seqlists = [ [] for i in range(self.nseq)]
        seqprob = {nodeid:np.zeros(self.seqlen) for nodeid in self.nodeids}

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
                seqprob[nodeid][site-1] = prob

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
        return (alignment, seqprob)

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

class _TreeTime:
    """Baseclass - do not instantiate"""

    def _parsetreefile(self, treefile):
        with pt.Nexustreefile(treefile) as tf:
            tree = tf.readtree()
            node2id = self._nodename_to_id(tree)
        return tree, node2id

    ###########################################################################################

    def _nodename_to_id(self, tree):
        name2id = {}
        for parent in tree.intnodes:
            for child in tree.children(parent):
                if child in tree.intnodes:
                    nodename = tree.getlabel(parent,child)
                else:
                    nodename = child
                name2id[nodename] = child
        # Add nodename for root (which was listed at end of treestring)
        # a bit hackish, since I know there is a nodename and a branchlen: NODE_0000000:0.00100
        rootname = re.sub(":[0-9]\.[0-9]+", "", tree.below_root)
        name2id[rootname] = tree.root
        return name2id

###################################################################################################
###################################################################################################

class TreeTimeSeq(_TreeTime):

    def __init__(self, ancseqfile, seq_treefile):
        self.tree, self.origseqname2id = self._parsetreefile(seq_treefile)
        self.alignment, self.name2id = self._parsealignfile(ancseqfile, self.tree, self.origseqname2id)

    ###########################################################################################

    def _parsealignfile(self, alignfile, tree, origname2id):
        """Reads alignment; returns alignment with nodes renamed to match nodes on seqtree"""
        sf = sequencelib.Seqfile(alignfile)
        alignment = sf.read_alignment()
        name2id = {}
        for origname in alignment.seqnamelist.copy():
            newid = origname2id[origname]
            newname = str(newid)
            name2id[newname] = newid
            alignment.changeseqname(origname, newname)
        return alignment, name2id

###################################################################################################
###################################################################################################

class TreeTimeTrait(_TreeTime):

    def __init__(self, trait_treefile, trait_probfile):
        self.tree, self.traitname2id = self._parsetreefile(trait_treefile)
        self.traitdict, self.traitprob = self._parsetraits(trait_treefile, trait_probfile,
                                                        self.tree, self.traitname2id)

    ###########################################################################################

    def _parsetraits(self, trait_treefile, trait_probfile, trait_tree, traitname2id):
        traitdict = self._parse_annot_tree(trait_treefile, traitname2id)
        traitprob, traitid2AB = self._parse_CSV(trait_probfile, traitname2id)
        traitdict = self._match_AB_trait(traitid2AB, traitdict, trait_tree)   # Add root trait. Check consistency
        return traitdict, traitprob

    ###########################################################################################

    def _parse_annot_tree(self, trait_treefile, traitname2id):
        with open(trait_treefile, "r") as infile:
            while True:
                line = infile.readline()
                if line.startswith("Begin Trees"):
                    break
            treestring = infile.readline()
            while True:
                line = infile.readline()
                if line.startswith("End;"):
                    break
                else:
                    treestring += line
        treestring = treestring.replace("\n", "")
        nodenames = re.findall(r"[\(\),]([^\(\),:]+):", treestring)
        traits = re.findall(r'\[&[\w]+=\"(\w+)\"\]', treestring)

        traitdict = {}
        for traitname, trait in zip(nodenames, traits):
            traitid = traitname2id[traitname]
            traitdict[traitid] = trait
        return traitdict

    ###########################################################################################

    def _parse_CSV(self, trait_probfile, traitname2id):
        traitprob = {}
        traitid2AB = {}
        with open(trait_probfile) as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                traitid = traitname2id[row[0]]
                pA, pB = float(row[1]), float(row[2])
                if pA > pB:
                    traitprob[traitid] = pA
                    traitid2AB[traitid] = "A"
                else:
                    traitprob[traitid] = pB
                    traitid2AB[traitid] = "B"
        return traitprob, traitid2AB

    ###########################################################################################

    def _match_AB_trait(self, traitid2AB, traitdict, trait_tree):
        # Python note: hardwired to two possible trait values. Think about generalizing
        random_leaf = trait_tree.leaflist()[0]
        ABval = traitid2AB[random_leaf]
        traitval = traitdict[random_leaf]
        possible_traitvals = set(traitdict.values())
        other_trait = (possible_traitvals - {traitval}).pop()
        AB2trait = {}
        if ABval == "A":
            AB2trait["A"] = traitval
            AB2trait["B"] = other_trait
        else:
            AB2trait["A"] = other_trait
            AB2trait["B"] = traitval
        self._check_trait_consistency(AB2trait, traitid2AB, traitdict)
        return traitdict

    ###########################################################################################

    def _check_trait_consistency(self, AB2trait, traitid2AB, traitdict):
        for traitid,traitval in traitdict.items():
            ABval = traitid2AB[traitid]
            expected_trait = AB2trait[ABval]
            if expected_trait != traitdict[traitid]:
                raise AncError(f"Inconsistency between confidence.csv and annotated tree for node {traitid}:\n"
                               f"Trait in CSV file:       {expected_trait}\n"
                               f"Trait in annotated tree: {traitdict[traitid]}")






