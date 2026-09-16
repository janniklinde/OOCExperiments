import it.unimi.dsi.webgraph.*;
import java.util.*;

public class TwitterTileProfile {
  static final int[] SIZES={1000,10000,20000,50000,100000,200000,400000};
  static long quant(TreeMap<Long,Long> h,long n,double q) {
    long target=(long)Math.ceil(n*q), seen=0;
    for(var e:h.entrySet()) {seen+=e.getValue(); if(seen>=target)return e.getKey();}
    return 0;
  }
  public static void main(String[] args)throws Exception {
    ImmutableGraph g=ImmutableGraph.loadOffline(args[0]); int n=g.numNodes();
    long[][] counts=new long[SIZES.length][];
    ArrayList<TreeMap<Long,Long>> hist=new ArrayList<>();
    for(int k=0;k<SIZES.length;k++){counts[k]=new long[(n+SIZES[k]-1)/SIZES[k]];hist.add(new TreeMap<>());}
    NodeIterator it=g.nodeIterator(); long edges=0; long start=System.nanoTime();
    while(it.hasNext()) {
      int row=it.nextInt(), deg=it.outdegree(); int[] a=it.successorArray();
      for(int j=0;j<deg;j++) counts[0][a[j]/1000]++;
      edges+=deg;
      if((row+1)%1000==0 || row+1==n) {
        for(int col=0;col<counts[0].length;col++) {
          long v=counts[0][col]; if(v==0)continue;
          hist.get(0).merge(v,1L,Long::sum);
          for(int k=1;k<SIZES.length;k++)counts[k][col/(SIZES[k]/1000)]+=v;
          counts[0][col]=0;
        }
        for(int k=1;k<SIZES.length;k++)if((row+1)%SIZES[k]==0 || row+1==n) {
          for(int c=0;c<counts[k].length;c++) {long v=counts[k][c];if(v>0)hist.get(k).merge(v,1L,Long::sum);counts[k][c]=0;}
        }
      }
      if((row+1)%2000000==0)System.err.printf("rows=%d edges=%d seconds=%.1f%n",row+1,edges,(System.nanoTime()-start)/1e9);
    }
    System.out.println("blocksize,total_tiles,nonempty_tiles,empty_pct,p50_nnz,p95_nnz,p99_nnz,max_nnz,tiles_under_1024nnz_pct,edges_in_under_1024nnz_pct,edges");
    for(int k=0;k<SIZES.length;k++) {
      var h=hist.get(k); long occupied=0,small=0,smallEdges=0,sum=0;
      for(var e:h.entrySet()){occupied+=e.getValue();sum+=e.getKey()*e.getValue();if(e.getKey()<1024){small+=e.getValue();smallEdges+=e.getKey()*e.getValue();}}
      if(sum!=edges)throw new AssertionError("edge mismatch "+sum);
      long width=counts[k].length,total=width*width;
      System.out.printf(Locale.ROOT,"%d,%d,%d,%.4f,%d,%d,%d,%d,%.4f,%.4f,%d%n",SIZES[k],total,occupied,100.0*(total-occupied)/total,quant(h,occupied,.5),quant(h,occupied,.95),quant(h,occupied,.99),h.lastKey(),100.0*small/occupied,100.0*smallEdges/edges,edges);
    }
  }
}
